///////////////////////////////////////////////////////////////
// CullInstancesPass - 实例剔除 Compute Shader
// 功能：对每个实例执行视锥剔除 + 遮挡剔除，
//       通过剔除的实例分为两类输出：
//       1. 正常绘制实例 → _drawnInstancesList
//       2. Billboard（屏幕占比过小）→ _drawnImpostorsList
///////////////////////////////////////////////////////////////

// 位掩码常量，用于从 flags 中提取对应字段
const b11 = 3u; // binary 0b11，2位掩码
const b111 = 7u; // binary 0b111，3位掩码
const b1111 = 15u; // binary 0b1111，4位掩码
const b11111 = 31u; // binary 0b11111，5位掩码
const b111111 = 63u; // binary 0b111111，6位掩码

///////////////////////////////////////////////////////////////
// Uniform 结构体 - 来自 CPU 端的全局渲染参数 (binding 0)
///////////////////////////////////////////////////////////////
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
  softwareRasterizerThreshold: f32, // 软件光栅化阈值
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
// 通用工具函数
///////////////////////////////////////////////////////////////

/// 计算 MVP 矩阵: projMat * viewMat * modelMat
fn getMVP_Mat(modelMat: mat4x4<f32>, viewMat: mat4x4<f32>, projMat: mat4x4<f32>) -> mat4x4<f32> {
  let a = viewMat * modelMat;
  return projMat * a;
}

/// 向上取整除法
fn ceilDivideU32(numerator: u32, denominator: u32) -> u32 {
  return (numerator + denominator - 1) / denominator;
}


///////////////////////////////////////////////////////////////
// 视锥剔除: 判断包围球是否在相机视锥内
///////////////////////////////////////////////////////////////
fn isInsideCameraFrustum(
  modelMat: mat4x4<f32>,
  boundingSphere: vec4f    // xyz=中心(模型空间), w=半径
) -> bool {
  // 将包围球中心从模型空间变换到世界空间
  var center = vec4f(boundingSphere.xyz, 1.);
  center = modelMat * center;
  let r = boundingSphere.w;
  // 判断世界空间球心到6个视锥平面的距离是否都 <= 半径
  // 如果球心在所有平面的内侧（距离 <= r），则球在视锥内
  let r0 = dot(center, _uniforms.cameraFrustumPlane0) <= r;
  let r1 = dot(center, _uniforms.cameraFrustumPlane1) <= r;
  let r2 = dot(center, _uniforms.cameraFrustumPlane2) <= r;
  let r3 = dot(center, _uniforms.cameraFrustumPlane3) <= r;
  let r4 = dot(center, _uniforms.cameraFrustumPlane4) <= r;
  let r5 = dot(center, _uniforms.cameraFrustumPlane5) <= r;
  return r0 && r1 && r2 && r3 && r4 && r5;
}


///////////////////////////////////////////////////////////////
// 深度相关工具
///////////////////////////////////////////////////////////////

/// 将 mip level 钳制到纹理的有效范围
fn clampToMipLevels(v: i32, _texture: texture_2d<f32>) -> i32 {
  let mipLevels = textureNumLevels(_texture);
  return clamp(v, 0, i32(mipLevels - 1));
}


/// 将深度缓冲区值 [0,1] 线性化为视图空间深度 [zNear, zFar]
fn linearizeDepth(depth: f32) -> f32 {
  let zNear: f32 = 0.01f;
  let zFar: f32 = 100f;
  
  // 推导过程:
  // 投影矩阵 PP:
  //   PP[10] = zFar / (zNear - zFar)
  //   PP[14] = (zFar * zNear) / (zNear - zFar)
  //   PP[11] = -1; PP[15] = 0; w = 1
  // z = PP[10] * p.z + PP[14] * w
  // w = PP[11] * p.z + PP[15] * w
  // z' = z / w = (PP[10] * p.z + PP[14]) / (-p.z)
  // 反解: p.z = (zFar * zNear) / (zFar + (zNear - zFar) * z')
  return zNear * zFar / (zFar + (zNear - zFar) * depth);
}

/// 将深度值线性化到 [0, 1] 范围
fn linearizeDepth_0_1(depth: f32) -> f32 {
  let zNear: f32 = 0.01f;
  let zFar: f32 = 100f;
  let d2 = linearizeDepth(depth);
  return d2 / (zFar - zNear);
}


/**
 * 近距离阈值: 离相机比这更近的物体总是通过遮挡剔除。
 * 修复包围球投影在靠近/穿过 zNear 时的 AABB 投影问题。
 * 大多数简单的球体投影公式在这种情况都不work。
 * 理论上可以在视图空间检测，如:
 *    'sphere.z < zNear && sphere.z + r > zNear'
 * 但会出现闪烁和不稳定。
 * 
 * 精确值选取: 因为我说是就是。
 */
const CLOSE_RANGE_NEAR_CAMERA: f32 = 4.0;

///////////////////////////////////////////////////////////////
// 遮挡剔除: 基于上一帧深度金字塔判断实例是否被遮挡
///////////////////////////////////////////////////////////////
/** 
 * 参考: 
 * https://www.youtube.com/live/Fj1E1A4CPCM?si=PJmBhKd_TQk1GMOb&t=2462 - triangles
 * https://www.youtube.com/watch?v=5sBpo5wKmEM - meshlets
*/
fn isPassingOcclusionCulling(
  modelMat: mat4x4<f32>,       // 实例模型矩阵
  boundingSphere: vec4f,        // 包围球 (xyz=中心, w=半径)
  dbgOverrideMipmapLevel: i32   // 调试: 覆盖 mipmap 层级 (>=0 时生效)
) -> bool {
  let viewportSize = _uniforms.viewport.xy;
  let viewMat = _uniforms.viewMatrix;
  let projMat = _uniforms.projMatrix;

  // 将包围球中心从模型空间变换到视图空间
  // 注意: 视图空间中 z 为负值（朝屏幕内部）
  let center = viewMat * modelMat * vec4f(boundingSphere.xyz, 1.);
  let r = boundingSphere.w;

  // 包围球上离相机最近的点的 z 值
  let closestPointZ = abs(center.z) - r;

  // 将球体投影到屏幕空间，获取 AABB (UV空间)
  var aabb = vec4f();
  let projectionOK = projectSphereView(projMat, center.xyz, r, &aabb);
  if (!projectionOK) { return true; } // 投影失败(球太近) → 总是可见

  // 计算球体在屏幕上的像素跨度
  let pixelSpanW = abs(aabb.z - aabb.x) * viewportSize.x;
  let pixelSpanH = abs(aabb.w - aabb.y) * viewportSize.y;
  let pixelSpan = max(pixelSpanW, pixelSpanH);

  // 计算 mipmap 层级:
  // 球体占50px → 上取整到64px → log2(64)=6 → 采样 mip6
  // 但深度金字塔第0层是半分辨率，所以额外 +1
  var mipLevel = i32(ceil(log2(pixelSpan)));
  if (dbgOverrideMipmapLevel >= 0) { mipLevel = dbgOverrideMipmapLevel; } // 调试覆盖
  mipLevel = clampToMipLevels(mipLevel + 1, _depthPyramidTexture);

  // 从深度金字塔采样，获取该位置最深的深度值
  let depthFromDepthBuffer = textureSampleLevel(_depthPyramidTexture, _depthSampler, aabb.xy, f32(mipLevel)).x;

  // 将采样深度线性化到视图空间，然后比较:
  // 如果球体最近点深度 <= 深度缓冲中的深度 → 球体可见(未被遮挡)
  let depthFromDepthBufferVS = linearizeDepth(depthFromDepthBuffer);
  return closestPointZ <= depthFromDepthBufferVS;
}

///////////////////////////////////////////////////////////////
// 球体投影: 将视图空间球体投影为屏幕 AABB
///////////////////////////////////////////////////////////////

/** 暴力法: 将包围球8个角点投影取AABB (备用方案，未启用) */
fn getAABBfrom8ProjectedPoints(projMat: mat4x4f, center: vec3f, r: f32) -> vec4f {
  let bb0 = getBB(projMat, center.xyz, r, vec3f( 1.,  1., 1.));
  let bb1 = getBB(projMat, center.xyz, r, vec3f(-1., -1., 1.));
  let bb2 = getBB(projMat, center.xyz, r, vec3f(-1.,  1., 1.));
  let bb3 = getBB(projMat, center.xyz, r, vec3f( 1., -1., 1.));
  let bb4 = getBB(projMat, center.xyz, r, vec3f( 1.,  1., -1.));
  let bb5 = getBB(projMat, center.xyz, r, vec3f(-1., -1., -1.));
  let bb6 = getBB(projMat, center.xyz, r, vec3f(-1.,  1., -1.));
  let bb7 = getBB(projMat, center.xyz, r, vec3f( 1., -1., -1.));
  // 取8个投影点的 x/y 极值，形成裁剪空间 AABB
  let aabbClip = vec4(
    min(min(min(bb0.x, bb1.x), min(bb2.x, bb3.x)), min(min(bb4.x, bb5.x), min(bb6.x, bb7.x))), // min x
    min(min(min(bb0.y, bb1.y), min(bb2.y, bb3.y)), min(min(bb4.y, bb5.y), min(bb6.y, bb7.y))), // min y
    max(max(max(bb0.x, bb1.x), max(bb2.x, bb3.x)), max(max(bb4.x, bb5.x), max(bb6.x, bb7.x))), // max x
    max(max(max(bb0.y, bb1.y), max(bb2.y, bb3.y)), max(max(bb4.y, bb5.y), max(bb6.y, bb7.y))), // max y
  );
  return (aabbClip + 1.0) * 0.5; // 从 [-1,1] 变换到 [0,1] UV 空间
}

/// 辅助: 投影球体上一个点到裁剪空间
fn getBB(projMat: mat4x4f, center: vec3f, r: f32, dir: vec3f) -> vec4f {
  let p = center + r * dir;
  let pProj = projMat * vec4f(p, 1.);
  return pProj / pProj.w; // 透视除法
}

/**
 * 解析法: 球体投影为紧凑 AABB (当前使用)
 * 参考: https://github.com/zeux/niagara/blob/master/src/shaders/math.h#L2
 * 论文: 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere. Mara & McGuire. 2013
 * 
 * @param projMat 投影矩阵
 * @param centerViewSpace 球心(视图空间)
 * @param r 球体半径
 * @param pixelSpan 输出: UV空间的 AABB (minX, minY, maxX, maxY)
 * @return true=投影成功, false=球太近无法投影(应视为可见)
 */
fn projectSphereView(
  projMat: mat4x4f,
  centerViewSpace: vec3f,
  r: f32,
  pixelSpan: ptr<function, vec4f>
) -> bool {
  let zNear: f32 = 0.01;
  // 如果球体离相机太近，投影公式不可靠，直接返回"可见"
  let closestPointZ = abs(centerViewSpace.z) - r;
  if (closestPointZ < zNear + CLOSE_RANGE_NEAR_CAMERA){
    return false; // 投影失败，调用方应视为可见
  }

  // 注意: 此算法仅适用于透视相机
  // 视图空间 z 取反是为了让 z 朝正方向(远离相机)
  let c = vec3f(centerViewSpace.xy, -centerViewSpace.z);
  let cr = c * r;
  let czr2 = c.z * c.z - r * r;

  // 解析求解球体投影的 X 方向边界
  let vx = sqrt(c.x * c.x + czr2);
  let minX = (vx * c.x - cr.z) / (vx * c.z + cr.x);
  let maxX = (vx * c.x + cr.z) / (vx * c.z - cr.x);

  // 解析求解球体投影的 Y 方向边界
  let vy = sqrt(c.y * c.y + czr2);
  let minY = (vy * c.y - cr.z) / (vy * c.z + cr.y);
  let maxY = (vy * c.y + cr.z) / (vy * c.z - cr.y);

  // 乘以投影矩阵的对角元素，从 NDC 转换到裁剪空间
  let P00 = projMat[0][0];
  let P11 = projMat[1][1];
  var aabb = vec4(minX * P00, minY * P11, maxX * P00, maxY * P11);
  // Y轴翻转 + 从 [-1,1] 变换到 [0,1] UV 空间
  aabb = aabb.xwzy * vec4(0.5, -0.5, 0.5, -0.5) + vec4(0.5);
  *pixelSpan = aabb;

  return true;
}

/**
 * 将模型空间包围球投影到屏幕，计算像素级跨度
 * 用于判断实例是否应降级为 Billboard
 */
fn projectSphereToScreen(
  modelMat: mat4x4<f32>,
  boundingSphere: vec4f,
  pixelSpan: ptr<function,vec2f>    // 输出: (宽像素, 高像素)
) -> bool {
  let viewportSize = _uniforms.viewport.xy;
  let viewMat = _uniforms.viewMatrix;
  let projMat = _uniforms.projMatrix;
  var aabb = vec4f();
  // 将球心变换到视图空间并投影
  let center = viewMat * modelMat * vec4f(boundingSphere.xyz, 1.);
  let r = boundingSphere.w;
  let projectionOK = projectSphereView(projMat, center.xyz, r, &aabb);
  // UV空间 AABB × 视口尺寸 = 像素跨度
  *pixelSpan = vec2f(
    abs(aabb.z - aabb.x) * viewportSize.x,
    abs(aabb.w - aabb.y) * viewportSize.y
  );
  return projectionOK;
}


///////////////////////////////////////////////////////////////
// Binding 1: 实例变换矩阵数组 (输入, 只读)
///////////////////////////////////////////////////////////////
@group(0) @binding(1)
var<storage, read> _instanceTransforms: array<mat4x4<f32>>;

fn _getInstanceTransform(idx: u32) -> mat4x4<f32> {
  return _instanceTransforms[idx];
}

fn _getInstanceCount() -> u32 {
  return arrayLength(&_instanceTransforms);
}


///////////////////////////////////////////////////////////////
// Binding 2: 已绘制实例的间接 dispatch 参数 (输出, 读写)
///////////////////////////////////////////////////////////////
/** 用于 dispatchWorkgroupsIndirect 的参数结构体
 *  https://developer.mozilla.org/en-US/docs/Web/API/GPUComputePassEncoder/dispatchWorkgroupsIndirect
 */
struct CullParams{
  workgroupsX: u32,               // 由 globalId=0 的线程设置
  workgroupsY: atomic<u32>,       // 原子递增: 每个可见实例+1，上限 32768
  workgroupsZ: u32,               // 固定为 1
  actuallyDrawnInstances: atomic<u32>, // 实际可见实例总数(不受 dispatch 限制)
  objectBoundingSphere: vec4f,    // 物体的包围球(模型空间)
  allMeshletsCount: u32,          // 物体包含的 meshlet 总数
}
@group(0) @binding(2)
var<storage, read_write> _drawnInstancesParams: CullParams;

///////////////////////////////////////////////////////////////
// Binding 3: 通过剔除的实例 ID 列表 (输出, 读写)
///////////////////////////////////////////////////////////////
@group(0) @binding(3)
var<storage, read_write> _drawnInstancesList: array<u32>;


///////////////////////////////////////////////////////////////
// Binding 4: Billboard 间接绘制参数 (输出, 读写)
///////////////////////////////////////////////////////////////
/** 用于 drawIndirect 的参数结构体
 *  https://developer.mozilla.org/en-US/docs/Web/API/GPUComputePassEncoder/dispatchWorkgroupsIndirect
 */
struct DrawIndirect{
  vertexCount: u32,           // 固定 6 (2个三角形组成的四边形)
  instanceCount: atomic<u32>, // 原子递增: 每个 Billboard 实例+1
  firstVertex: u32,           // 固定 0
  firstInstance : u32,        // 固定 0
}
@group(0) @binding(4)
var<storage, read_write> _drawnImpostorsParams: DrawIndirect;

///////////////////////////////////////////////////////////////
// Binding 5: Billboard 实例 ID 列表 (输出, 读写)
///////////////////////////////////////////////////////////////
@group(0) @binding(5)
var<storage, read_write> _drawnImpostorsList: array<u32>;


///////////////////////////////////////////////////////////////
// Binding 6-7: 上一帧深度金字塔纹理 + 采样器 (输入, 只读)
///////////////////////////////////////////////////////////////
@group(0) @binding(6)
var _depthPyramidTexture: texture_2d<f32>;
@group(0) @binding(7)
var _depthSampler: sampler;


///////////////////////////////////////////////////////////////
// 主入口: Compute Shader
// workgroup 大小: 32×1×1 = 每个工作组 32 个线程
// dispatch 数量由 CPU 端 min(instancesCount, 32768) 决定
///////////////////////////////////////////////////////////////
@compute
@workgroup_size(32, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  // 首先初始化间接绘制参数(仅 global_id.x==0 的线程执行)
  // 必须放在最前面，防止后续 early return 导致参数未被设置
  resetOtherDrawParams(global_id);

  let settingsFlags = _uniforms.flags;
  let boundingSphere = _drawnInstancesParams.objectBoundingSphere; // 物体的包围球
  let MAX_WORKGROUPS_Y: u32 = 32768u; // GPU dispatch 的 Y 维度上限

  // 将实例分配给各线程:
  // 如果实例数 > 32768 (dispatch上限)，则每个线程需处理多个实例
  let instanceCount: u32 = _getInstanceCount();
  let iterCount: u32 = ceilDivideU32(instanceCount, 32768u); // 每个线程需处理的实例数
  let tfxOffset: u32 = global_id.x * iterCount; // 该线程负责的起始实例索引

  // 遍历分配给当前线程的所有实例
  for(var i: u32 = 0u; i < iterCount; i++){
    let tfxIdx: u32 = tfxOffset + i;
    if (tfxIdx >= instanceCount) { continue; } // 越界保护
    let modelMat = _getInstanceTransform(tfxIdx);

    // 步骤1: 判断实例是否可见 (视锥剔除 + 遮挡剔除)
    if (!isInstanceRendered(settingsFlags, modelMat, boundingSphere)){
      continue; // 被剔除，跳过
    }

    // 步骤2: 判断是否应降级为 Billboard (屏幕占比过小)
    if (renderAsBillboard(settingsFlags, modelMat, boundingSphere)) {
      // 写入 Billboard 列表
      let idx = atomicAdd(&_drawnImpostorsParams.instanceCount, 1u);
      _drawnImpostorsList[idx] = tfxIdx;

    } else {
      // 写入正常绘制列表
      // workgroupsY 原子递增，用于后续 CullMeshletsPass 的间接 dispatch
      // 但不能超过 MAX_WORKGROUPS_Y 上限
      atomicAdd(&_drawnInstancesParams.workgroupsY, 1u);
      atomicMin(&_drawnInstancesParams.workgroupsY, MAX_WORKGROUPS_Y);
      
      // 实际可见实例计数(不受 dispatch 限制)
      let idx = atomicAdd(&_drawnInstancesParams.actuallyDrawnInstances, 1u);
      _drawnInstancesList[idx] = tfxIdx;
    }
  } 
}

///////////////////////////////////////////////////////////////
// 工具函数: 判断实例是否应该渲染
///////////////////////////////////////////////////////////////
fn isInstanceRendered(
  settingsFlags: u32,
  modelMat: mat4x4<f32>,
  boundingSphere: vec4f
) -> bool {
  // 视锥剔除: 包围球是否在相机视锥内
  if (
    useInstancesFrustumCulling(settingsFlags) &&
    !isInsideCameraFrustum(modelMat, boundingSphere)
  ) {
    return false;
  }

  // 遮挡剔除: 包围球是否被深度金字塔中的物体遮挡
  let overrideMipmap = getOverrideOcclusionCullMipmap(settingsFlags);
  if (
    useInstancesOcclusionCulling(settingsFlags) &&
    !isPassingOcclusionCulling(modelMat, boundingSphere, overrideMipmap)
  ) {
    return false;
  }

  return true;
}


///////////////////////////////////////////////////////////////
// 工具函数: 判断实例是否应渲染为 Billboard
///////////////////////////////////////////////////////////////
fn renderAsBillboard(
  settingsFlags: u32,
  modelMat: mat4x4<f32>,
  boundingSphere: vec4f
) -> bool {
  // 强制 Billboard 模式
  if (useForceBillboards(settingsFlags)) {
    return true;
  }

  // 计算包围球在屏幕上的像素面积
  var pixelSpan = vec2f();
  let projectionOK = projectSphereToScreen(modelMat, boundingSphere, &pixelSpan);
  // 如果投影成功 且 像素面积 < 阈值 → 降级为 Billboard
  return (
    projectionOK &&
    pixelSpan.x * pixelSpan.y < _uniforms.billboardThreshold
  );
}

///////////////////////////////////////////////////////////////
// 初始化间接绘制参数 (仅 global_id.x==0 的线程执行)
///////////////////////////////////////////////////////////////
fn resetOtherDrawParams(global_id: vec3<u32>){
  if (global_id.x == 0u) {
    // CullMeshletsPass 的 dispatch 参数:
    // workgroupsX = ceil(allMeshletsCount / 32)，因为 meshlet 剔除的 workgroup 大小为 32
    _drawnInstancesParams.workgroupsX = ceilDivideU32(
      _drawnInstancesParams.allMeshletsCount,
      32u
    );
    _drawnInstancesParams.workgroupsZ = 1u;

    // Billboard 绘制参数:
    // 每个 Billboard 由 2个三角形(6个顶点)组成的四边形
    _drawnImpostorsParams.vertexCount = 6u;
    _drawnImpostorsParams.firstVertex = 0u;
    _drawnImpostorsParams.firstInstance = 0u;
  }
}
