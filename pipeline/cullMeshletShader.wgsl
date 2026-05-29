///////////////////////////////////////////////////////////////
// CullMeshletsPass - Meshlet 剔除 Compute Shader
// 功能：对每个 meshlet × 可见实例 执行三重判定：
//   1. 视锥剔除 (Frustum Culling)
//   2. 遮挡剔除 (Occlusion Culling，基于上一帧深度金字塔)
//   3. LOD 误差判定 (Nanite 核心：选择合适的 LOD 级别)
// 通过判定的 meshlet 根据屏幕大小分流到：
//   - 软件光栅化队列 (小三角形)
//   - 硬件光栅化队列 (大三角形)
///////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////
// 通用工具函数
///////////////////////////////////////////////////////////////

/// 计算 MVP 矩阵: projMat * viewMat * modelMat
/// 将坐标从模型空间直接变换到裁剪空间
fn getMVP_Mat(modelMat: mat4x4<f32>, viewMat: mat4x4<f32>, projMat: mat4x4<f32>) -> mat4x4<f32> {
  let a = viewMat * modelMat;
  return projMat * a;
}

/// 向上取整除法: ceil(numerator / denominator)
/// 用于计算 dispatch 的工作组数量或每个线程需处理的迭代次数
fn ceilDivideU32(numerator: u32, denominator: u32) -> u32 {
  return (numerator + denominator - 1) / denominator;
}


///////////////////////////////////////////////////////////////
// 深度相关工具函数
///////////////////////////////////////////////////////////////

/// 将 mip level 钳制到纹理的有效范围
/// 例如 8 个 mip levels 对应索引 0-7
fn clampToMipLevels(v: i32, _texture: texture_2d<f32>) -> i32 {
  let mipLevels = textureNumLevels(_texture);
  return clamp(v, 0, i32(mipLevels - 1));
}


/// 将深度缓冲区非线性深度值 [0,1] 线性化为视图空间深度 [zNear, zFar]
/// 
/// 推导过程:
///   投影矩阵 PP 中:
///     PP[2][2] = zFar / (zNear - zFar)
///     PP[3][2] = (zFar * zNear) / (zNear - zFar)
///     PP[2][3] = -1; PP[3][3] = 0; w = 1
///   
///   z_clip = PP[2][2] * p.z + PP[3][2] * w
///   w_clip = PP[2][3] * p.z + PP[3][3] * w = -p.z
///   z_ndc = z_clip / w_clip = (PP[2][2] * p.z + PP[3][2]) / (-p.z)
///   
///   反解 p.z:
///     p.z = (zFar * zNear) / (zFar + (zNear - zFar) * z_ndc)
fn linearizeDepth(depth: f32) -> f32 {
  let zNear: f32 = 0.01f;
  let zFar: f32 = 100f;
  return zNear * zFar / (zFar + (zNear - zFar) * depth);
}

/// 将深度值线性化到 [0, 1] 范围 (0=zNear, 1=zFar)
fn linearizeDepth_0_1(depth: f32) -> f32 {
  let zNear: f32 = 0.01f;
  let zFar: f32 = 100f;
  let d2 = linearizeDepth(depth);
  return d2 / (zFar - zNear);
}


///////////////////////////////////////////////////////////////
// 遮挡剔除: 基于上一帧深度金字塔判断包围球是否被遮挡
///////////////////////////////////////////////////////////////

/**
 * 近距离阈值常量: 离相机比这更近的物体总是通过遮挡剔除。
 * 
 * 原因: 当包围球靠近/穿过近裁剪面时，球体投影公式数学上不稳定，
 * 会导致 AABB 计算结果抖动。简单地在视图空间检测
 * "sphere.z < zNear && sphere.z + r > zNear" 会产生闪烁。
 * 
 * 所以直接用经验值: 距离近裁剪面 4 个单位以内的物体一律视为可见。
 */
const CLOSE_RANGE_NEAR_CAMERA: f32 = 4.0;

/** 
 * 遮挡剔除核心函数
 * 
 * 算法流程:
 *   1. 将包围球变换到视图空间
 *   2. 用解析公式将球体投影到屏幕 AABB
 *   3. 根据 AABB 像素大小选择深度金字塔的 mipmap 层级
 *   4. 采样深度金字塔获取该位置最深的已有深度
 *   5. 比较球体最近点深度与已有深度，判断是否被遮挡
 * 
 * 参考:
 *   https://www.youtube.com/live/Fj1E1A4CPCM?si=PJmBhKd_TQk1GMOb&t=2462 - triangles
 *   https://www.youtube.com/watch?v=5sBpo5wKmEM - meshlets
 */
fn isPassingOcclusionCulling(
  modelMat: mat4x4<f32>,       // 实例模型矩阵
  boundingSphere: vec4f,        // 包围球 (xyz=中心模型空间, w=半径)
  dbgOverrideMipmapLevel: i32   // 调试: 覆盖 mipmap 层级 (>=0 时生效)
) -> bool {
  let viewportSize = _uniforms.viewport.xy;
  let viewMat = _uniforms.viewMatrix;
  let projMat = _uniforms.projMatrix;

  // 步骤1: 将包围球中心从模型空间 → 视图空间
  // 注意: 视图空间中 z 为负值（朝屏幕内部）
  let center = viewMat * modelMat * vec4f(boundingSphere.xyz, 1.);
  let r = boundingSphere.w;

  // 包围球上离相机最近的点的深度值
  let closestPointZ = abs(center.z) - r;

  // 步骤2: 将球体解析投影到屏幕 AABB (UV空间 [0,1])
  var aabb = vec4f(); // (minX, minY, maxX, maxY) in UV space
  let projectionOK = projectSphereView(projMat, center.xyz, r, &aabb);
  if (!projectionOK) { 
    return true; // 投影失败(球太靠近近裁剪面) → 总是可见
  }

  // 步骤3: 计算 AABB 在全屏下的像素跨度
  let pixelSpanW = abs(aabb.z - aabb.x) * viewportSize.x;
  let pixelSpanH = abs(aabb.w - aabb.y) * viewportSize.y;
  let pixelSpan = max(pixelSpanW, pixelSpanH); // 取最大维度

  // 步骤4: 根据 AABB 像素大小计算需要采样的 mipmap 层级
  // 原理: meshlet 占 50px → 上取整到 64px → log2(64)=6 → 采样 mip6
  // 但深度金字塔第0层是半分辨率，所以额外 +1
  var mipLevel = i32(ceil(log2(pixelSpan)));
  if (dbgOverrideMipmapLevel >= 0) { 
    mipLevel = dbgOverrideMipmapLevel; // 调试模式: 强制使用指定层级
  }
  mipLevel = clampToMipLevels(mipLevel + 1, _depthPyramidTexture);

  // 步骤5: 从深度金字塔采样，获取该位置已有的最远深度值
  // 使用 aabb.xy (AABB中心) 作为采样坐标
  // textureSampleLevel 可指定 mipmap 层级进行采样
  let depthFromDepthBuffer = textureSampleLevel(
    _depthPyramidTexture, _depthSampler, aabb.xy, f32(mipLevel)
  ).x;

  // 步骤6: 比较深度
  // 将深度缓冲值线性化到视图空间，然后与球体最近点深度比较
  // 如果球体最近点深度 <= 深度缓冲中的深度 → 球体未被遮挡 → 可见
  let depthFromDepthBufferVS = linearizeDepth(depthFromDepthBuffer);
  return closestPointZ <= depthFromDepthBufferVS;
}

///////////////////////////////////////////////////////////////
// 球体投影: 将视图空间球体投影为屏幕 AABB
///////////////////////////////////////////////////////////////

/** 暴力法: 将包围球8个角点投影到裁剪空间取 AABB (备用方案，当前未使用)
 * 
 * 思路: 沿8个对角方向偏移 r 的点投影后取极值
 * 缺点: 8次投影 + 大量 min/max，比解析法慢
 */
fn getAABBfrom8ProjectedPoints(projMat: mat4x4f, center: vec3f, r: f32) -> vec4f {
  let bb0 = getBB(projMat, center.xyz, r, vec3f( 1.,  1., 1.));
  let bb1 = getBB(projMat, center.xyz, r, vec3f(-1., -1., 1.));
  let bb2 = getBB(projMat, center.xyz, r, vec3f(-1.,  1., 1.));
  let bb3 = getBB(projMat, center.xyz, r, vec3f( 1., -1., 1.));
  let bb4 = getBB(projMat, center.xyz, r, vec3f( 1.,  1., -1.));
  let bb5 = getBB(projMat, center.xyz, r, vec3f(-1., -1., -1.));
  let bb6 = getBB(projMat, center.xyz, r, vec3f(-1.,  1., -1.));
  let bb7 = getBB(projMat, center.xyz, r, vec3f( 1., -1., -1.));
  // 8个投影点取 x/y 极值，形成裁剪空间 [-1,1] 的 AABB
  let aabbClip = vec4(
    min(min(min(bb0.x, bb1.x), min(bb2.x, bb3.x)), min(min(bb4.x, bb5.x), min(bb6.x, bb7.x))), // min x
    min(min(min(bb0.y, bb1.y), min(bb2.y, bb3.y)), min(min(bb4.y, bb5.y), min(bb6.y, bb7.y))), // min y
    max(max(max(bb0.x, bb1.x), max(bb2.x, bb3.x)), max(max(bb4.x, bb5.x), max(bb6.x, bb7.x))), // max x
    max(max(max(bb0.y, bb1.y), max(bb2.y, bb3.y)), max(max(bb4.y, bb5.y), max(bb6.y, bb7.y))), // max y
  );
  return (aabbClip + 1.0) * 0.5; // 从 [-1,1] 变换到 [0,1] UV 空间
}

/// 辅助函数: 将球体上沿 dir 方向偏移 r 的点投影到裁剪空间
fn getBB(projMat: mat4x4f, center: vec3f, r: f32, dir: vec3f) -> vec4f {
  let p = center + r * dir;
  let pProj = projMat * vec4f(p, 1.);
  return pProj / pProj.w; // 透视除法
}

/**
 * 解析法: 将视图空间球体投影为紧凑的屏幕 AABB (当前使用)
 * 
 * 论文: "2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere"
 *        Michael Mara, Morgan McGuire. 2013
 * 代码参考: https://github.com/zeux/niagara/blob/master/src/shaders/math.h#L2
 * 博客: https://zeux.io/2023/01/12/approximate-projected-bounds/
 * 
 * 相比暴力8点法，解析法只需一次 sqrt + 除法即可得到精确的投影边界，
 * 且结果更紧凑（不会过度保守）。
 * 
 * @param projMat        投影矩阵
 * @param centerViewSpace 球心(视图空间)
 * @param r              球体半径
 * @param pixelSpan      输出: UV空间的 AABB (minX, minY, maxX, maxY)
 * @return true=投影成功, false=球太近无法投影(应视为可见)
 * 
 * 注意: 此算法仅适用于透视相机
 */
fn projectSphereView(
  projMat: mat4x4f,
  centerViewSpace: vec3f,
  r: f32,
  pixelSpan: ptr<function, vec4f>
) -> bool {
  let zNear: f32 = 0.01;
  // 如果球体离相机太近，投影公式不可靠，直接返回"投影失败"(=可见)
  let closestPointZ = abs(centerViewSpace.z) - r;
  if (closestPointZ < zNear + CLOSE_RANGE_NEAR_CAMERA){
    return false;
  }

  // 视图空间 z 取反，让 z 朝正方向(远离相机)
  // 因为视图空间中 z 为负值，解析公式需要正值
  let c = vec3f(centerViewSpace.xy, -centerViewSpace.z);
  let cr = c * r;
  let czr2 = c.z * c.z - r * r;

  // 解析求解球体投影的 X 方向边界
  // 原理: 球体在透视投影下的轮廓是二次曲线，解析解可得精确的 min/max
  let vx = sqrt(c.x * c.x + czr2);
  let minX = (vx * c.x - cr.z) / (vx * c.z + cr.x);
  let maxX = (vx * c.x + cr.z) / (vx * c.z - cr.x);

  // 解析求解球体投影的 Y 方向边界
  let vy = sqrt(c.y * c.y + czr2);
  let minY = (vy * c.y - cr.z) / (vy * c.z + cr.y);
  let maxY = (vy * c.y + cr.z) / (vy * c.z - cr.y);

  // 乘以投影矩阵的对角元素 P00、P11，从 NDC 变换到裁剪空间
  // P00 = 1/(aspect*tan(fov/2)), P11 = 1/tan(fov/2)
  let P00 = projMat[0][0];
  let P11 = projMat[1][1];
  var aabb = vec4(minX * P00, minY * P11, maxX * P00, maxY * P11);
  // Y轴翻转 + 从 [-1,1] 变换到 [0,1] UV 空间
  // xwzy swizzle: 交换 Y 分量方向; *0.5+0.5: 映射到 [0,1]
  aabb = aabb.xwzy * vec4(0.5, -0.5, 0.5, -0.5) + vec4(0.5);
  *pixelSpan = aabb;

  return true;
}

/**
 * 将模型空间包围球投影到屏幕，计算像素级跨度
 * 用于判断 meshlet 应走软件光栅化还是硬件光栅化
 * 
 * @param modelMat       模型矩阵
 * @param boundingSphere 包围球 (xyz=中心, w=半径)
 * @param pixelSpan      输出: (宽像素数, 高像素数)
 * @return true=投影成功, false=球太近(应视为可见)
 */
fn projectSphereToScreen(
  modelMat: mat4x4<f32>,
  boundingSphere: vec4f,
  pixelSpan: ptr<function,vec2f>
) -> bool {
  let viewportSize = _uniforms.viewport.xy;
  let viewMat = _uniforms.viewMatrix;
  let projMat = _uniforms.projMatrix;
  var aabb = vec4f();
  // 将球心变换到视图空间并投影
  let center = viewMat * modelMat * vec4f(boundingSphere.xyz, 1.);
  let r = boundingSphere.w;
  let projectionOK = projectSphereView(projMat, center.xyz, r, &aabb);
  // UV空间 AABB × 视口像素尺寸 = 像素跨度
  *pixelSpan = vec2f(
    abs(aabb.z - aabb.x) * viewportSize.x,
    abs(aabb.w - aabb.y) * viewportSize.y
  );
  return projectionOK;
}


///////////////////////////////////////////////////////////////
// 视锥剔除: 判断包围球是否在相机视锥体内
///////////////////////////////////////////////////////////////

/**
 * 将包围球中心变换到世界空间，与6个视锥裁剪平面比较。
 * 球心到某平面距离 <= 半径 → 该平面测试通过。
 * 6个平面全部通过 → 球在视锥内(含部分在内的情况)。
 * 
 * 这是保守策略: 部分在视锥内的球也会通过，避免误剔除。
 */
fn isInsideCameraFrustum(
  modelMat: mat4x4<f32>,
  boundingSphere: vec4f    // xyz=中心(模型空间), w=半径
) -> bool {
  // 将包围球中心从模型空间变换到世界空间
  var center = vec4f(boundingSphere.xyz, 1.);
  center = modelMat * center;
  let r = boundingSphere.w;
  // 判断世界空间球心到6个视锥平面的有符号距离是否 <= 半径
  // dot(center, plane) = 球心到平面的有符号距离
  //   负值 = 在平面内侧(视锥内)
  //   正值 = 在平面外侧
  //   <= r = 球体与平面相交或在内侧
  let r0 = dot(center, _uniforms.cameraFrustumPlane0) <= r;
  let r1 = dot(center, _uniforms.cameraFrustumPlane1) <= r;
  let r2 = dot(center, _uniforms.cameraFrustumPlane2) <= r;
  let r3 = dot(center, _uniforms.cameraFrustumPlane3) <= r;
  let r4 = dot(center, _uniforms.cameraFrustumPlane4) <= r;
  let r5 = dot(center, _uniforms.cameraFrustumPlane5) <= r;
  return r0 && r1 && r2 && r3 && r4 && r5;
}


///////////////////////////////////////////////////////////////
// LOD 误差判定: Nanite 核心 —— 选择合适的 LOD 级别
///////////////////////////////////////////////////////////////

/**
 * 判断 meshlet 是否处于"正确的 LOD 级别"
 * 
 * Nanite 的 meshlet 组织成 LOD 树结构:
 *   - 叶节点 = 最高精度(最细)
 *   - 根节点 = 最低精度(最粗)
 *   - 每组兄弟节点由同一个父节点简化而来
 * 
 * 判定规则(与 Epic Nanite 演讲一致):
 *   parentError > threshold  AND  clusterError <= threshold
 *   → 父节点太粗糙(需要更精细) + 当前节点够精细 → 渲染此节点
 * 
 *   parentError <= threshold → 父节点够精细 → 不渲染此节点(用父节点代替)
 *   clusterError > threshold → 当前节点也太粗糙 → 不渲染此节点(需渲染子节点)
 */
fn isCorrectNaniteLOD (
  modelMat: mat4x4<f32>,
  meshlet: NaniteMeshletTreeNode
) -> bool {
  let flags = _uniforms.flags;

  let threshold = _uniforms.viewport.z;   // 误差阈值(像素)，默认 0.5
  let screenHeight = _uniforms.viewport.y; // 视口高度(像素)
  let cotHalfFov = _uniforms.viewport.w;   // cot(fov/2)，用于投影误差计算
  let mvpMatrix = getMVP_Mat(modelMat, _uniforms.viewMatrix, _uniforms.projMatrix);

  // 当前节点的简化误差(与兄弟节点的最大简化误差)
  let clusterError = getProjectedError(
    mvpMatrix,
    screenHeight,
    cotHalfFov,
    meshlet.boundsMidPointAndError,
  );
  // 父节点的简化误差
  let parentError = getProjectedError(
    mvpMatrix,
    screenHeight,
    cotHalfFov,
    meshlet.parentBoundsMidPointAndError,
  );

  // 核心: 父节点误差 > 阈值(父太粗) 且 当前节点误差 <= 阈值(当前够精)
  return parentError > threshold && clusterError <= threshold;
}


/**
 * 计算屏幕空间投影误差(像素)
 * 
 * 将世界空间的简化误差转换为屏幕像素误差。
 * 公式来源: Federico Ponchio 论文
 * "Multiresolution structures for interactive visualization of very large 3D datasets"
 * 
 * 中的屏幕空间误差饱和公式。
 * 
 * 直观理解:
 *   - cotHalfFov: 视场角一半的余切 (FOV越小，值越大，同等距离下像素误差越大)
 *   - r: 世界空间简化误差(简化掉的几何距离)
 *   - d2: meshlet到相机的距离平方(裁剪空间)
 *   - 结果: 简化误差在屏幕上对应多少像素
 * 
 * 当 meshlet 距离越远 → 像素误差越小 → 可使用更粗糙的 LOD
 * 当 meshlet 距离越近 → 像素误差越大 → 需要更精细的 LOD
 */
fn getProjectedError(
  mvpMatrix: mat4x4<f32>,
  screenHeight: f32,
  cotHalfFov: f32,
  boundsMidPointAndError: vec4f   // xyz=包围球中心(世界空间), w=简化误差
) -> f32 {
  let r = boundsMidPointAndError.w; // 世界空间简化误差
  
  // 根节点的父节点误差为"无穷大"(GPU上用极大值 99990.0 代替)
  // 直接返回，避免后续 sqrt(d2 - r*r) 出现负数
  if (r >= PARENT_ERROR_INFINITY) {
    return PARENT_ERROR_INFINITY;
  }

  // 将包围球中心变换到裁剪空间
  let center = mvpMatrix * vec4f(boundsMidPointAndError.xyz, 1.0f);
  // 到相机的距离平方(裁剪空间近似)
  let d2 = dot(center.xyz, center.xyz);
  // 投影半径公式: (cotHalfFov * r) / sqrt(d2 - r*r)
  // 这是球体透视投影的屏幕空间半径近似
  let projectedR = (cotHalfFov * r) / sqrt(d2 - r * r);
  // 转换为像素误差
  return (projectedR * screenHeight) / 2.0;
}


///////////////////////////////////////////////////////////////
// Uniform 结构体 - 来自 CPU 端的全局渲染参数 (binding 0)
///////////////////////////////////////////////////////////////

// 位掩码常量，用于从 flags 中提取对应字段
const b11 = 3u; // binary 0b11，2位掩码
const b111 = 7u; // binary 0b111，3位掩码
const b1111 = 15u; // binary 0b1111，4位掩码
const b11111 = 31u; // binary 0b11111，5位掩码
const b111111 = 63u; // binary 0b111111，6位掩码

struct Uniforms {
  vpMatrix: mat4x4<f32>,       // 视图-投影组合矩阵（世界空间 → 裁剪空间）
  vpMatrixInv: mat4x4<f32>,    // vpMatrix 的逆矩阵（裁剪空间 → 世界空间）
  viewMatrix: mat4x4<f32>,     // 视图矩阵（世界空间 → 相机空间）
  projMatrix: mat4x4<f32>,     // 投影矩阵（相机空间 → 裁剪空间）
  viewport: vec4f,             // (width, height, errorThreshold, cotHalfFov)
  cameraPosition: vec4f,       // 相机世界坐标 (xyz, padding)
  cameraFrustumPlane0: vec4f,  // 视锥平面0 (左)
  cameraFrustumPlane1: vec4f,  // 视锥平面1 (右)
  cameraFrustumPlane2: vec4f,  // 视锥平面2 (下)
  cameraFrustumPlane3: vec4f,  // 视锥平面3 (上)
  cameraFrustumPlane4: vec4f,  // 视锥平面4 (近)
  cameraFrustumPlane5: vec4f,  // 视锥平面5 (远)
  // flags 位字段说明:
  // b1   - meshlet 视锥剔除开关
  // b2   - meshlet 遮挡剔除开关
  // b3,4,5 - 着色模式 (1 << 2 to 1 << 4)
  // b6   - 实例视锥剔除开关
  // b7   - 实例遮挡剔除开关
  // b8,9,10,11 - 调试: 深度金字塔显示层级 (0-15)
  // b12,13,14,15 - 调试: 覆盖遮挡剔除 mipmap 层级 (0-15). 0b1111 = 关闭
  // b16  - 强制所有实例渲染为 Billboard
  // b17..b22 - Billboard 抖动强度
  // b23..b32 - 未使用
  flags: u32,
  billboardThreshold: f32,          // Billboard 阈值: 屏幕像素面积小于此值时降级为 Billboard
  softwareRasterizerThreshold: f32, // 软件光栅化阈值: 屏幕像素面积小于此值时走软件光栅化
  padding0: u32,
  colorMgmt: vec4f,                 // (gamma, exposure, ditherStrength, padding)
};
@binding(0) @group(0)
var<uniform> _uniforms: Uniforms;

///////////////////////////////////////////////////////////////
// Flags 解析函数
///////////////////////////////////////////////////////////////
fn checkFlag(flags: u32, bit: u32) -> bool { return (flags & bit) > 0; }
fn useFrustumCulling(flags: u32) -> bool { return checkFlag(flags, 1u); }           // bit 0: meshlet 视锥剔除
fn useOcclusionCulling(flags: u32) -> bool { return checkFlag(flags, 2u); }         // bit 1: meshlet 遮挡剔除
fn useInstancesFrustumCulling(flags: u32) -> bool { return checkFlag(flags, 32u); } // bit 5: 实例视锥剔除
fn useInstancesOcclusionCulling(flags: u32) -> bool { return checkFlag(flags, 64u); } // bit 6: 实例遮挡剔除
fn useForceBillboards(flags: u32) -> bool { return checkFlag(flags, 65536u); }      // bit 16: 强制 Billboard
fn getShadingMode(flags: u32) -> u32 {
  return (flags >> 2u) & b111; // bits 2-4: 着色模式
}
fn getDbgPyramidMipmapLevel(flags: u32) -> i32 {
  return i32(clamp((flags >> 8u) & b1111, 0u, 15u)); // bits 8-11: 调试金字塔层级
}
fn getOverrideOcclusionCullMipmap(flags: u32) -> i32 {
  let v: u32 = clamp((flags >> 12u) & b1111, 0u, 15u); // bits 12-15: 覆盖遮挡 mipmap
  if (v == 15u) { return -1; } // 0b1111 表示关闭覆盖
  return i32(v);
}
fn getBillboardDitheringStrength(flags: u32) -> f32 {
  let v: u32 = (flags >> 17u) & b111111; // bits 17-22: [0-63] 映射到 [0.0, 1.0]
  return f32(v) / 63.0;
}


///////////////////////////////////////////////////////////////
// Meshlet 数据结构 (binding 1)
///////////////////////////////////////////////////////////////

/**
 * Nanite Meshlet LOD 树节点
 * 
 * 每个节点存储:
 *   - 自身包围球 (用于视锥/遮挡剔除)
 *   - 兄弟共享包围球 + 最大简化误差 (用于 LOD 判定)
 *   - 父节点包围球 + 父节点误差 (用于 LOD 判定)
 *   - 渲染所需的索引数据
 */
struct NaniteMeshletTreeNode {
  boundsMidPointAndError: vec4f, // 共享兄弟包围球中心(xyz) + maxSiblingsError(w)
  parentBoundsMidPointAndError: vec4f, // 父节点包围球中心(xyz) + parentError(w)
  ownBoundingSphere: vec4f, // 该 meshlet 自身的包围球 (用于视锥/遮挡剔除)
  triangleCount: u32,       // 该 meshlet 包含的三角形数量
  firstIndexOffset: u32,    // 索引缓冲区中的起始偏移
  lodLevel: u32,            // LOD 层级 (0=最精细)
  padding0: u32,            // 对齐填充
}
@group(0) @binding(1)
var<storage, read> _meshlets: array<NaniteMeshletTreeNode>;


///////////////////////////////////////////////////////////////
// 硬件光栅化间接绘制参数 (binding 3, 输出)
///////////////////////////////////////////////////////////////

/** 用于 drawIndirect 的参数结构体
 *  传给 GPU 的 drawIndirect 调用，实现无需 CPU 回读的绘制
 */
struct DrawIndirect{
  vertexCount: u32,           // 顶点数 = 128 * 3 (每个 meshlet 最多 128 三角形)
  instanceCount: atomic<u32>, // 原子递增: 每个通过的 meshlet +1
  firstVertex: u32,           // 固定 0
  firstInstance : u32,        // 固定 0
}
@group(0) @binding(3)
var<storage, read_write> _drawnMeshletsParams: DrawIndirect;


///////////////////////////////////////////////////////////////
// 软件光栅化间接 dispatch 参数 (binding 4, 输出)
///////////////////////////////////////////////////////////////

/** 用于 dispatchWorkgroupsIndirect 的参数结构体
 *  小三角形走软件光栅化 (compute shader 模拟光栅化)
 */
struct DrawnMeshletsSw{
  workgroupsX: u32,               // 由 globalId=0 的线程设置 = ceil(128/32) = 4
  workgroupsY: atomic<u32>,       // 原子递增: 每个软件光栅化 meshlet +1，上限 32768
  workgroupsZ: u32,               // 固定 1
  actuallyDrawnMeshlets: atomic<u32>, // 实际软件光栅化的 meshlet 总数(不受上限限制)
}
@group(0) @binding(4)
var<storage, read_write> _drawnMeshletsSwParams: DrawnMeshletsSw;


///////////////////////////////////////////////////////////////
// 已绘制 meshlet 列表 (binding 5, 输出)
///////////////////////////////////////////////////////////////

// 每个元素是 vec2u(tfxIdx, meshletIdx)
// - tfxIdx: 实例索引
// - meshletIdx: meshlet 索引
@group(0) @binding(5)
var<storage, read_write> _drawnMeshletsList: array<vec2<u32>>;

/// 硬件光栅化: 从列表头部开始存储
fn _storeMeshletHardwareDraw(idx: u32, tfxIdx: u32, meshletIdx: u32) {
  _drawnMeshletsList[idx] = vec2u(tfxIdx, meshletIdx);
}
/// 软件光栅化: 从列表尾部倒序存储
/// 避免硬件和软件光栅化的写入冲突
fn _storeMeshletSoftwareDraw(idx: u32, tfxIdx: u32, meshletIdx: u32) {
  let len: u32 = arrayLength(&_drawnMeshletsList);
  let idx2: u32 = len - 1u - idx; // 列表末尾倒序
  _drawnMeshletsList[idx2] = vec2u(tfxIdx, meshletIdx);
}


///////////////////////////////////////////////////////////////
// 实例变换矩阵 (binding 2, 输入)
///////////////////////////////////////////////////////////////
@group(0) @binding(2)
var<storage, read> _instanceTransforms: array<mat4x4<f32>>;

fn _getInstanceTransform(idx: u32) -> mat4x4<f32> {
  return _instanceTransforms[idx];
}

fn _getInstanceCount() -> u32 {
  return arrayLength(&_instanceTransforms);
}


///////////////////////////////////////////////////////////////
// 深度金字塔纹理 + 采样器 (binding 6-7, 输入)
///////////////////////////////////////////////////////////////
@group(0) @binding(6)
var _depthPyramidTexture: texture_2d<f32>;
@group(0) @binding(7)
var _depthSampler: sampler;

/** GPU 端替代 Infinity 的值
 * JS 端用 errorValue=Infinity 表示根节点无父节点，
 * 但 GPU 传输 Infinity 有风险，所以用极大值 99990.0 代替
 */
const PARENT_ERROR_INFINITY: f32 = 99990.0f;


///////////////////////////////////////////////////////////////
// SHADER VARIANT 1: 使用 Y/Z 维度编码实例 ID
///////////////////////////////////////////////////////////////

/**
 * dispatch 方式: dispatchWorkgroups(meshletCount, min(instanceCount, 32768), ceil(instanceCount/32768))
 * 
 * 实例 ID 编码: tfxIdx = global_id.z * 32768 + global_id.y
 * 
 * 优点: 逻辑简单，每个线程只处理一个 (meshlet, instance) 对
 * 缺点: 当实例数远小于 32768 时，大量线程空转浪费
 */
@compute
@workgroup_size(32, 1, 1)
fn main_SpreadYZ(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  // 初始化间接绘制参数(仅 global_id.x==0 的线程执行)
  resetOtherDrawParams(global_id);

  // X 维度 = meshlet 索引
  let meshletIdx: u32 = global_id.x;
  if (meshletIdx >= arrayLength(&_meshlets)) {
    return; // 超出 meshlet 数量，空转
  }
  let meshlet = _meshlets[meshletIdx];

  // 从 Y/Z 维度重建实例索引
  // Y 维度范围 [0, 32767]，Z 维度用于超过 32768 的实例
  let tfxIdx: u32 = (global_id.z * 32768u) + global_id.y;
  if (tfxIdx >= _getInstanceCount()) {
    return; // 超出实例数量，空转
  }
  let modelMat = _getInstanceTransform(tfxIdx);

  // 三重判定: 视锥 + 遮挡 + LOD
  let settingsFlags = _uniforms.flags;
  if (isMeshletRendered(settingsFlags, modelMat, meshlet)){
    registerDraw(modelMat, meshlet.ownBoundingSphere, tfxIdx, meshletIdx);
  }
}


///////////////////////////////////////////////////////////////
// SHADER VARIANT 2: 在 shader 内迭代实例
///////////////////////////////////////////////////////////////

/**
 * dispatch 方式: dispatchWorkgroups(meshletCount, 1, 1)
 * 
 * 每个线程处理一个 meshlet，在 shader 内部循环遍历所有实例。
 * 当实例数 > 32768 时，每个线程需处理多个实例(iterCount > 1)。
 * 
 * 优点: 比 Variant 1 更少的空线程
 * 缺点: 仍遍历所有实例(包括被 CullInstancesPass 剔除的)
 */
@compute
@workgroup_size(32, 1, 1)
fn main_Iter(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  // 初始化间接绘制参数
  resetOtherDrawParams(global_id);

  // X 维度 = meshlet 索引
  let meshletIdx: u32 = global_id.x;
  if (meshletIdx >= arrayLength(&_meshlets)) {
    return;
  }
  let meshlet = _meshlets[meshletIdx];
  let settingsFlags = _uniforms.flags;

  // 计算每个线程需处理的实例数
  // 除以 32768 是因为 dispatch Y 维度上限为 32768
  // 但此变体只 dispatch 1 个 Y，所以所有实例由 X 维度的线程分担
  let instanceCount: u32 = _getInstanceCount();
  let iterCount: u32 = ceilDivideU32(instanceCount, 32768u);
  let tfxOffset: u32 = global_id.y * iterCount;

  // 遍历分配给当前线程的实例
  for(var i: u32 = 0u; i < iterCount; i++){
    let tfxIdx: u32 = tfxOffset + i;
    let modelMat = _getInstanceTransform(tfxIdx);

    if (isMeshletRendered(settingsFlags, modelMat, meshlet)){
      registerDraw(modelMat, meshlet.ownBoundingSphere, tfxIdx, meshletIdx);
    }
  } 
}


///////////////////////////////////////////////////////////////
// SHADER VARIANT 3: 间接 dispatch，仅遍历已通过实例剔除的实例
///////////////////////////////////////////////////////////////

/**
 * 这是最高效的变体，默认启用。
 * 
 * dispatch 方式: dispatchWorkgroupsIndirect(_drawnInstancesParams)
 *   workgroupsX = ceil(allMeshletsCount / 32) (由 CullInstancesPass 设置)
 *   workgroupsY = 可见实例数 (由 CullInstancesPass 原子递增)
 *   workgroupsZ = 1
 * 
 * 优点: 
 *   - 只遍历 CullInstancesPass 通过的实例，跳过已剔除的
 *   - 避免在已剔除实例上浪费 GPU 线程
 * 缺点:
 *   - 需要额外的 binding (8, 9) 读取实例剔除结果
 */

// 来自 CullInstancesPass 的输出 (binding 8, 只读)
/** 间接 dispatch 参数，由 CullInstancesPass 写入 */
struct CullParams{
  workgroupsX: u32,               // 由 CullInstancesPass 的 globalId=0 线程设置
  workgroupsY: u32,               // 可见实例数(非 atomic，CullInstancesPass 已完成写入)
  workgroupsZ: u32,               // 固定 1
  actuallyDrawnInstances: u32,    // 实际可见实例总数
  objectBoundingSphere: vec4f,    // 物体包围球
  allMeshletsCount: u32,          // meshlet 总数
}
@group(0) @binding(8)
var<storage, read> _drawnInstancesParams: CullParams;

// 来自 CullInstancesPass 的可见实例 ID 列表 (binding 9, 只读)
@group(0) @binding(9)
var<storage, read> _drawnInstancesList: array<u32>;


@compute
@workgroup_size(32, 1, 1)
fn main_Indirect(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  // 初始化间接绘制参数
  resetOtherDrawParams(global_id);

  // X 维度 = meshlet 索引
  let meshletIdx: u32 = global_id.x;
  if (meshletIdx >= arrayLength(&_meshlets)) {
    return;
  }
  let meshlet = _meshlets[meshletIdx];
  let settingsFlags = _uniforms.flags;

  // 从 CullInstancesPass 的结果中获取可见实例数量
  let instanceCount: u32 = _drawnInstancesParams.actuallyDrawnInstances;
  // 计算每个线程需处理的实例数
  let iterCount: u32 = ceilDivideU32(instanceCount, 32768u);
  let tfxOffset: u32 = global_id.y * iterCount;

  // 遍历可见实例
  for(var i: u32 = 0u; i < iterCount; i++){
    let iterOffset: u32 = tfxOffset + i;
    // 从可见实例列表中获取实例 ID(而非全部实例)
    let tfxIdx: u32 = _drawnInstancesList[iterOffset];
    let modelMat = _getInstanceTransform(tfxIdx);

    if (isMeshletRendered(settingsFlags, modelMat, meshlet)){
      registerDraw(modelMat, meshlet.ownBoundingSphere, tfxIdx, meshletIdx);
    }
  } 
}


///////////////////////////////////////////////////////////////
// 工具函数: 三重判定入口
///////////////////////////////////////////////////////////////

/**
 * 判断 meshlet 是否应该被渲染
 * 依次执行:
 *   1. 视锥剔除 — 包围球是否在相机视锥内
 *   2. 遮挡剔除 — 包围球是否被深度金字塔中的几何体遮挡
 *   3. LOD 误差判定 — 当前 meshlet 是否处于正确的 LOD 层级
 */
fn isMeshletRendered(
  settingsFlags: u32,
  modelMat: mat4x4<f32>,
  meshlet: NaniteMeshletTreeNode
) -> bool {
  // 1. 视锥剔除: 使用 meshlet 自身包围球
  if (
    useFrustumCulling(settingsFlags) &&
    !isInsideCameraFrustum(modelMat, meshlet.ownBoundingSphere)
  ) {
    return false;
  }

  // 2. 遮挡剔除: 使用 meshlet 自身包围球 + 上一帧深度金字塔
  let overrideMipmap = getOverrideOcclusionCullMipmap(settingsFlags);
  if (
    useOcclusionCulling(settingsFlags) &&
    !isPassingOcclusionCulling(modelMat, meshlet.ownBoundingSphere, overrideMipmap)
  ) {
    return false;
  }

  // 3. LOD 误差判定: Nanite 核心
  return isCorrectNaniteLOD(modelMat, meshlet);
}

///////////////////////////////////////////////////////////////
// 初始化间接绘制参数 (仅 global_id.x==0 的线程执行)
///////////////////////////////////////////////////////////////

/**
 * 设置硬件/软件光栅化的间接绘制/ dispatch 参数
 * 这些参数只需设置一次，所以由 global_id.x==0 的线程负责
 */
fn resetOtherDrawParams(global_id: vec3<u32>){
  if (global_id.x == 0u) {
    // 硬件光栅化参数:
    // 每个 meshlet 最多 128 个三角形，每个三角形 3 个顶点
    // 实际绘制时，超出 triangleCount 的顶点会被 discard
    _drawnMeshletsParams.vertexCount = 128u * 3u;
    _drawnMeshletsParams.firstVertex = 0u;
    _drawnMeshletsParams.firstInstance = 0u;

    // 软件光栅化参数:
    // workgroupsX = ceil(128 / 32) = 4
    // (软件光栅化的 workgroup 大小为 32，每个 meshlet 最多 128 三角形)
    _drawnMeshletsSwParams.workgroupsX = ceilDivideU32(128u, 32u);
    _drawnMeshletsSwParams.workgroupsZ = 1u;
  }
}

///////////////////////////////////////////////////////////////
// 注册绘制: 将通过的 meshlet 加入绘制队列
///////////////////////////////////////////////////////////////

/**
 * 根据屏幕大小将 meshlet 分流到软件/硬件光栅化队列
 * 
 * 分流策略:
 *   - 屏幕面积 < softwareRasterizerThreshold (默认 1360 像素)
 *     → 软件光栅化 (小三角形用 compute shader 模拟更高效)
 *   - 否则 → 硬件光栅化 (大三角形用传统 rasterizer)
 * 
 * TODO [LOW]: 可用 ballot 操作优化原子写入
 * 参考: Wihlidal 2015 "Optimizing the Graphics Pipeline with Compute"
 * 思路: warp 内多个线程同时 atomicAdd 时，可以先在 warp 内求和，
 * 再一次性写入全局计数器，然后重新分配索引给各线程。
 */
fn registerDraw(
  modelMat: mat4x4<f32>,
  boundingSphere: vec4f,
  tfxIdx: u32,       // 实例索引
  meshletIdx: u32    // meshlet 索引
){
  // 计算 meshlet 在屏幕上的像素面积
  var pixelSpan = vec2f();
  let projectionOK = projectSphereToScreen(modelMat, boundingSphere, &pixelSpan);
  
  // 判断是否走软件光栅化: 投影成功且屏幕面积 < 阈值
  let useSoftwareRasterizer = projectionOK &&
    pixelSpan.x * pixelSpan.y < _uniforms.softwareRasterizerThreshold; 

  if (useSoftwareRasterizer) {
    // 软件光栅化路径:
    // workgroupsY 原子递增(用于间接 dispatch)，上限 32768
    let MAX_WORKGROUPS_Y: u32 = 32768u;
    atomicAdd(&_drawnMeshletsSwParams.workgroupsY, 1u);
    atomicMin(&_drawnMeshletsSwParams.workgroupsY, MAX_WORKGROUPS_Y);
      
    // 实际计数(不受上限限制)
    let idx = atomicAdd(&_drawnMeshletsSwParams.actuallyDrawnMeshlets, 1u);
    // 存入列表尾部(倒序，避免与硬件光栅化的头部写入冲突)
    _storeMeshletSoftwareDraw(idx, tfxIdx, meshletIdx);

  } else {
    // 硬件光栅化路径:
    // instanceCount 原子递增(用于间接 draw)
    let idx = atomicAdd(&_drawnMeshletsParams.instanceCount, 1u);
    // 存入列表头部(正序)
    _storeMeshletHardwareDraw(idx, tfxIdx, meshletIdx);
  }
}
