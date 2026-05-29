///////////////////////////////////////////////////////////////
// 软件光栅化 Shader (Software Rasterizer)
//
// 渲染管线中的位置:
//   CullInstancesPass → CullMeshletsPass → 硬件光栅化
//                                           └→ 软件光栅化 (本 shader)
//
// 软件光栅化的 meshlet 分流条件 (在 CullMeshletsPass 中决定):
//   - 三角形数 ≤ softwareRasterizerThreshold → 软件光栅化
//   - 三角形数 >  softwareRasterizerThreshold → 硬件光栅化
//
// 为什么需要软件光栅化?
//   小 meshlet (三角形数少) 用硬件光栅化效率低: 
//   每个 instance 画 MAX_MESHLET_TRIANGLES*3 顶点, GPU fixed-function 开销大。
//   软件光栅化在 compute shader 中逐三角形 rasterize，跳过流水线冗余。
//
// 核心算法:
//   1. 获取三角形顶点 → NDC → 屏幕像素坐标
//   2. 背面剔除 (CCW)
//   3. 计算包围盒 (bounding box) + scissor 裁剪
//   4. 逐像素遍历: 用增量式 edge function 判断像素是否在三角形内
//   5. 计算重心坐标 → 插值深度/法线
//   6. 写入 32bit payload (depth + encoded normal) → atomicMax 深度测试
//
// 参考: https://www.sctheblog.com/blog/hair-software-rasterize/
///////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////
// 工具函数
///////////////////////////////////////////////////////////////

/**
 * MVP 矩阵乘法: Projection × View × Model
 */
fn getMVP_Mat(modelMat: mat4x4<f32>, viewMat: mat4x4<f32>, projMat: mat4x4<f32>) -> mat4x4<f32> {
  let a = viewMat * modelMat;
  return projMat * a;
}


/** WGSL 的整数向上取整除法: ceil(a/b) */
fn ceilDivideU32(numerator: u32, denominator: u32) -> u32 {
  return (numerator + denominator - 1) / denominator;
}


/**
 * 将法线从模型空间变换到世界空间
 *
 * WARNING: 仅当模型矩阵无缩放时正确
 */
fn transformNormalToWorldSpace(modelMat: mat4x4f, normalV: vec3f) -> vec3f {
  let normalMatrix = modelMat;
  let normalWS = normalMatrix * vec4f(normalV, 0.0);
  return normalize(normalWS.xyz);
}


///////////////////////////////////////////////////////////////
// Octahedron 法线编码/解码
// 将 3D 单位法线压缩为 2D float, 节省存储
///////////////////////////////////////////////////////////////

fn OctWrap(v: vec2f) -> vec2f {
  let signX = select(-1.0, 1.0, v.x >= 0.0);
  let signY = select(-1.0, 1.0, v.y >= 0.0);
  return (1.0 - abs(v.yx)) * vec2f(signX, signY);
}

fn encodeOctahedronNormal(n0: vec3f) -> vec2f {
  var n = n0 / (abs(n0.x) + abs(n0.y) + abs(n0.z));
  if (n.z < 0.0) {
    let a = OctWrap(n.xy);
    n.x = a.x;
    n.y = a.y;
  }
  return n.xy * 0.5 + 0.5;
}

fn decodeOctahedronNormal(f_: vec2f) -> vec3f {
  let f = f_ * 2.0 - 1.0;
  var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
  let t = saturate(-n.z);
  if (n.x >= 0.0){ n.x -= t; } else { n.x += t; }
  if (n.y >= 0.0){ n.y -= t; } else { n.y += t; }
  return normalize(n);
}


///////////////////////////////////////////////////////////////
// 深度线性化
//
// WebGPU 默认使用反向 Z (Reverse Z, far=0, near=1),
// 且深度值在 NDC 中是非线性的。
// 这些函数将 non-linear NDC depth 还原为线性深度，
// 用于正确比较和写入。
///////////////////////////////////////////////////////////////

/**
 * 将 NDC 深度还原为 [zNear, zFar] 范围的线性深度
 * 推导:
 *   投影矩阵:
 *     PP[10] = zFar / (zNear - zFar)
 *     PP[14] = (zFar * zNear) / (zNear - zFar)
 *     PP[11] = -1, PP[15] = 0
 *   z_clip = PP[10]*p.z + PP[14]*w, w_clip = PP[11]*p.z
 *   z_ndc = z_clip / w_clip
 *   p.z = (zFar * zNear) / (zFar + (zNear - zFar) * z_ndc)
 */
fn linearizeDepth(depth: f32) -> f32 {
  let zNear: f32 = 0.01f;
  let zFar: f32 = 100f;
  return zNear * zFar / (zFar + (zNear - zFar) * depth);
}

/** 线性深度归一化到 [0, 1] */
fn linearizeDepth_0_1(depth: f32) -> f32 {
  let zNear: f32 = 0.01f;
  let zFar: f32 = 100f;
  let d2 = linearizeDepth(depth);
  return d2 / (zFar - zNear);
}


///////////////////////////////////////////////////////////////
// Uniforms & Flags
///////////////////////////////////////////////////////////////

    /// 位掩码常量
    const b11 = 3u;
    const b111 = 7u;
    const b1111 = 15u;
    const b11111 = 31u;
    const b111111 = 63u;

    /**
     * 全局 Uniform 结构体 (与所有 Pass 共享同一布局)
     *
     * flags 位字段:
     *   bit 1        - meshlet 视锥剔除
     *   bit 2        - meshlet 遮挡剔除
     *   bits 3-5     - 着色模式
     *   bits 6-7     - 实例级剔除
     *   bits 8-11    - 调试深度金字塔 mipmap
     *   bits 12-15   - 调试遮挡剔除 mipmap 覆盖
     *   bit 16       - 强制 billboard
     *   bits 17-22   - billboard 抖动强度
     *   bits 23-32   - 未使用
     */
    struct Uniforms {
      vpMatrix: mat4x4<f32>,
      vpMatrixInv: mat4x4<f32>,
      viewMatrix: mat4x4<f32>,
      projMatrix: mat4x4<f32>,
      viewport: vec4f,
      cameraPosition: vec4f,
      cameraFrustumPlane0: vec4f,
      cameraFrustumPlane1: vec4f,
      cameraFrustumPlane2: vec4f,
      cameraFrustumPlane3: vec4f,
      cameraFrustumPlane4: vec4f,
      cameraFrustumPlane5: vec4f,
      flags: u32,
      billboardThreshold: f32,
      softwareRasterizerThreshold: f32,
      padding0: u32,
      colorMgmt: vec4f,
    };

    @binding(0) @group(0)
    var<uniform> _uniforms: Uniforms;

    fn checkFlag(flags: u32, bit: u32) -> bool { return (flags & bit) > 0; }
    fn useFrustumCulling(flags: u32) -> bool { return checkFlag(flags, 1u); }
    fn useOcclusionCulling(flags: u32) -> bool { return checkFlag(flags, 2u); }
    fn useInstancesFrustumCulling(flags: u32) -> bool { return checkFlag(flags, 32u); }
    fn useInstancesOcclusionCulling(flags: u32) -> bool { return checkFlag(flags, 64u); }
    fn useForceBillboards(flags: u32) -> bool { return checkFlag(flags, 65536u); }
    fn getShadingMode(flags: u32) -> u32 {
      return (flags >> 2u) & b111;
    }
    fn getDbgPyramidMipmapLevel(flags: u32) -> i32 {
      return i32(clamp((flags >> 8u) & b1111, 0u, 15u));
    }
    fn getOverrideOcclusionCullMipmap(flags: u32) -> i32 {
      let v: u32 = clamp((flags >> 12u) & b1111, 0u, 15u);
      if (v == 15u) { return -1; }
      return i32(v);
    }
    fn getBillboardDitheringStrength(flags: u32) -> f32 {
      let v: u32 = (flags >> 17u) & b111111;
      return f32(v) / 63.0;
    }


///////////////////////////////////////////////////////////////
// Meshlet 数据结构 & GPU Buffer Bindings
///////////////////////////////////////////////////////////////

/**
 * Nanite Meshlet 树节点
 */
struct NaniteMeshletTreeNode {
  boundsMidPointAndError: vec4f,        // .xyz = 兄弟合并包围球, .w = clusterError
  parentBoundsMidPointAndError: vec4f,  // .xyz = 父包围球, .w = parentError
  ownBoundingSphere: vec4f,             // .xyz = 自身包围球, .w = 半径
  triangleCount: u32,                   // 三角形数
  firstIndexOffset: u32,                // 索引缓冲区偏移
  lodLevel: u32,                        // LOD 层级
  padding0: u32,
}

/// Binding 4: Meshlet 树节点数组 (只读)
@group(0) @binding(4)
var<storage, read> _meshlets: array<NaniteMeshletTreeNode>;


/**
 * 软件光栅化 dispatch 间接参数结构体
 *
 * 用作 dispatchWorkgroupsIndirect 的参数:
 *   - 前 3 个 u32 = (workgroupsX, workgroupsY, workgroupsZ)
 *   - 第 4 个 u32 = actuallyDrawnMeshlets (实际需绘制的 meshlet 总数)
 *
 * 注意: 与硬件光栅化的 CullParams 不同, 这里 workgroupsY 代表
 * meshlet 维度 (而非实例维度), 因为软件光栅化的每个线程
 * global_id.y 对应一个 meshlet 实例
 */
struct DrawnMeshletsSw{
  workgroupsX: u32,  // 三角形维度工作组数 (ceil(maxTriangles/32))
  workgroupsY: u32,  // meshlet 维度工作组数 (ceil(drawnMeshlets/32768))
  workgroupsZ: u32,  // 固定为 1
  actuallyDrawnMeshlets: u32,  // 实际通过剔除的软件光栅化 meshlet 总数
}

/// Binding 6: 软件光栅化 dispatch 参数 (只读)
@group(0) @binding(6)
var<storage, read> _drawnMeshletsSwParams: DrawnMeshletsSw;


/**
 * Binding 5: 通过剔除的 meshlet 绘制列表
 *
 * 每个元素是 vec2u:
 *   .x = instanceIdx (实例变换索引)
 *   .y = meshletIdx  (meshlet 索引)
 *
 * 与硬件光栅化共享同一 buffer:
 *   - 硬件: 从头部正序写入/读取
 *   - 软件: 从尾部倒序写入/读取
 */
@group(0) @binding(5)
var<storage, read> _drawnMeshletsList: array<vec2<u32>>;

fn _getMeshletHardwareDraw(idx: u32) -> vec2u {
  return _drawnMeshletsList[idx];
}
fn _getMeshletSoftwareDraw(idx: u32) -> vec2u {
  let len: u32 = arrayLength(&_drawnMeshletsList);
  let idx2: u32 = len - 1u - idx;
  return _drawnMeshletsList[idx2];
}


/**
 * Binding 2: 顶点位置缓冲区
 *
 * WARNING: 必须用 array<vec4f>! array<vec3f> 在 WGSL 中不工作.
 * 调试此问题花了大量时间.
 */
@group(0) @binding(2)
var<storage, read> _vertexPositionsNative: array<vec4f>;

fn _getVertexPosition(idx: u32) -> vec4f { return _vertexPositionsNative[idx]; }


/// Binding 8: 顶点法线 (octahedron 编码)
@group(0) @binding(8)
var<storage, read> _vertexNormals: array<vec2f>;

fn _getVertexNormal(idx: u32) -> vec3f {
  return decodeOctahedronNormal(_vertexNormals[idx]);
}


/// Binding 3: 索引缓冲区
@group(0) @binding(3)
var<storage, read> _indexBuffer: array<u32>;


/// Binding 7: 实例变换矩阵
@group(0) @binding(7)
var<storage, read> _instanceTransforms: array<mat4x4<f32>>;

fn _getInstanceTransform(idx: u32) -> mat4x4<f32> {
  return _instanceTransforms[idx];
}

fn _getInstanceCount() -> u32 {
  return arrayLength(&_instanceTransforms);
}


/**
 * Binding 1: 软件光栅化输出缓冲区 (核心!)
 *
 * 这是一个 atomic<u32> 数组, 每个像素一个元素.
 * 每个元素编码为 32bit payload:
 *   bits 16-31: depth (u16, reversed, 大的值 = 更近)
 *   bits 8-15:  normal.x (octahedron 编码, 0-255)
 *   bits 0-7:   normal.y (octahedron 编码, 0-255)
 *
 * 使用 atomicMax 进行深度比较:
 *   因为 depth 被反转 (1.0 - depth_ndc), 大的值 = 更近,
 *   所以 atomicMax 自然实现了"保留最近像素"的深度测试.
 *
 * 类比: 这相当于 color attachment + depth attachment,
 * 但由于 WGSL 缺少 atomic<u64>, 不得不将 depth+normal 压缩到 32bit.
 */
@group(0) @binding(1)
var<storage, read_write> _softwareRasterizerResult: array<atomic<u32>>;


///////////////////////////////////////////////////////////////
// 调试颜色 (ABGR 格式, 与 WebGPU 纹理布局一致)
///////////////////////////////////////////////////////////////

const COLOR_RED: u32    = 0xff0000ffu;  // ABGR: R=255
const COLOR_GREEN: u32  = 0xff00ff00u;  // ABGR: G=255
const COLOR_BLUE: u32   = 0xffff0000u;  // ABGR: B=255
const COLOR_TEAL: u32   = 0xffffff00u;  // ABGR: G+B=255
const COLOR_PINK: u32   = 0xffff00ffu;  // ABGR: R+B=255
const COLOR_YELLOW: u32 = 0xff00ffffu;  // ABGR: R+G=255


///////////////////////////////////////////////////////////////
// 主入口: Compute Shader
//
// 线程分配:
//   global_id.x = 三角形在 meshlet 内的索引 [0, MAX_TRIANGLES-1]
//   global_id.y = 软件光栅化 meshlet 列表中的索引
//   global_id.z = 1
//
// 一个线程处理一个 (三角形, meshlet) 对
// 当 meshlet 数量 > 32768 时, 每个线程迭代多个 meshlet
///////////////////////////////////////////////////////////////

@compute
@workgroup_size(32, 1, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let viewportSize: vec2f = _uniforms.viewport.xy;
  let viewMatrix = _uniforms.viewMatrix;
  let projMatrix = _uniforms.projMatrix;

  // 当前线程负责的三角形在 meshlet 内的索引
  let triangleIdx: u32 = global_id.x;

  // 软件光栅化 meshlet 总数
  let drawnMeshletCnt: u32 = _drawnMeshletsSwParams.actuallyDrawnMeshlets;
  // 当 meshlet 数 > 32768 时, 每个线程需处理多个 meshlet
  let iterCount: u32 = ceilDivideU32(drawnMeshletCnt, 32768u);
  let tfxOffset: u32 = global_id.y * iterCount;

  // 遍历当前线程负责的所有 meshlet
  for(var i: u32 = 0u; i < iterCount; i++){
    let iterOffset: u32 = tfxOffset + i;
    if (iterOffset >= drawnMeshletCnt) { continue; }

    // 获取 meshlet 数据
    let drawData: vec2u = _getMeshletSoftwareDraw(iterOffset);
    let meshlet = _meshlets[drawData.y];
    // 如果 meshlet 的三角形数 <= current triangle index, 跳过
    if (triangleIdx >= meshlet.triangleCount) { continue; }

    // 获取实例变换矩阵
    let modelMat = _getInstanceTransform(drawData.x);
    let mvpMat = getMVP_Mat(modelMat, viewMatrix, projMatrix);

    // 光栅化当前三角形
    let indexOffset = meshlet.firstIndexOffset;
    rasterize(
      modelMat,
      mvpMat,
      viewportSize,
      indexOffset,
      triangleIdx
    );
  }
}


///////////////////////////////////////////////////////////////
// 核心: 三角形光栅化
//
// 算法步骤:
//   1. 读取 3 个顶点 → 投影到 NDC → 映射到屏幕像素
//   2. 背面剔除
//   3. 计算 2D 包围盒, 裁剪到视口
//   4. 逐像素遍历, 增量式 edge function 判断像素覆盖
//   5. 重心坐标插值深度/法线
//   6. 编码 payload 并存储
///////////////////////////////////////////////////////////////

fn rasterize(
  modelMat: mat4x4f,
  mvpMat: mat4x4f,
  viewportSizeF32: vec2f,
  indexOffset: u32,
  triangleIdx: u32
) {
  let viewportSize = vec2u(viewportSizeF32);

  // 1. 读取 3 个顶点索引 (CCW 顺序)
  let idx0 = _indexBuffer[indexOffset + triangleIdx * 3u];
  let idx1 = _indexBuffer[indexOffset + triangleIdx * 3u + 1u];
  let idx2 = _indexBuffer[indexOffset + triangleIdx * 3u + 2u];
  let vertexPos0 = _getVertexPosition(idx0);
  let vertexPos1 = _getVertexPosition(idx1);
  let vertexPos2 = _getVertexPosition(idx2);

  // 投影到 NDC
  let v0_NDC: vec3f = projectVertex(mvpMat, vertexPos0);
  let v1_NDC: vec3f = projectVertex(mvpMat, vertexPos1);
  let v2_NDC: vec3f = projectVertex(mvpMat, vertexPos2);

  // NDC → 屏幕像素坐标
  let v0: vec2f = ndc2viewportPx(viewportSizeF32.xy, v0_NDC);
  let v1: vec2f = ndc2viewportPx(viewportSizeF32.xy, v1_NDC);
  let v2: vec2f = ndc2viewportPx(viewportSizeF32.xy, v2_NDC);

  // 法线处理
  let vertexN0 = _getVertexNormal(idx0);
  let vertexN1 = _getVertexNormal(idx1);
  let vertexN2 = _getVertexNormal(idx2);
  let n0 = transformNormalToWorldSpace(modelMat, vertexN0);
  let n1 = transformNormalToWorldSpace(modelMat, vertexN1);
  let n2 = transformNormalToWorldSpace(modelMat, vertexN2);

  // 2. 背面剔除
  // edgeFunction 返回有符号三角形面积的 2 倍
  // WebGPU 默认 CCW 为正, 所以取负值
  let triangleArea = -edgeFunction(v0, v1, v2);
  if (triangleArea < 0.) { return; }  // CW 三角形, 背面, 丢弃

  // 3. 计算 2D 包围盒 + scissor 裁剪到视口
  var boundRectMax: vec2f = ceil(max(max(v0, v1), v2));  // 右上
  var boundRectMin: vec2f = floor(min(min(v0, v1), v2)); // 左下
  boundRectMax = min(boundRectMax, viewportSizeF32.xy);  // 裁剪到视口
  boundRectMin = max(boundRectMin, vec2f(0.0, 0.0));

  // 4. 预计算增量式 edge function 系数
  // EdgeC 结构体: A*x + B*y + C
  // 当沿 x 方向遍历时, y 不变, 只需递加 A
  // 当换行时, 递加 B
  let CC0 = edgeC(v2, v1);  // 边 v2→v1, 用于顶点 0 的重心坐标
  let CC1 = edgeC(v0, v2);  // 边 v0→v2, 用于顶点 1 的重心坐标
  let CC2 = edgeC(v1, v0);  // 边 v1→v0, 用于顶点 2 的重心坐标

  // 采样点位于像素中心 (+0.5)
  // 参考: https://www.sctheblog.com/blog/hair-software-rasterize/#half-of-the-pixel-offset
  let firstSamplePoint = boundRectMin.xy + vec2f(0.5);

  // 初始化增量值 (firstSamplePoint 处的 edge function 值)
  var CY0 = firstSamplePoint.x * CC0.A + firstSamplePoint.y * CC0.B + CC0.C;
  var CY1 = firstSamplePoint.x * CC1.A + firstSamplePoint.y * CC1.B + CC1.C;
  var CY2 = firstSamplePoint.x * CC2.A + firstSamplePoint.y * CC2.B + CC2.C;
  let triangleArea2 = CY0 + CY1 + CY2;  // 验证: 应与上面一致

  // 5. 逐行遍历 (row-by-row)
  for (var y: f32 = boundRectMin.y; y < boundRectMax.y; y+=1.0) {
    // 每行开始时, 重置该行起点的 edge function 值
    var CX0 = CY0;
    var CX1 = CY1;
    var CX2 = CY2;

    // 逐列遍历 (column-by-column)
    for (var x: f32 = boundRectMin.x; x < boundRectMax.x; x+=1.0) {
      // 增量式 edge function 判断:
      // CX >= 0 表示该像素在边的内侧 (CCW 三角形)
      if (CX0 >= 0 && CX1 >= 0 && CX2 >= 0) {
        // 计算重心坐标
        let C0 = CX0 / triangleArea2;  // 顶点 0 的权重
        let C1 = CX1 / triangleArea2;  // 顶点 1 的权重
        let C2 = CX2 / triangleArea2;  // 顶点 2 的权重

        // 重心坐标插值: 深度和法线
        let depth: f32 = v0_NDC.z * C0 + v1_NDC.z * C1 + v2_NDC.z * C2;
        let n: vec3f = normalize(n0 * C0 + n1 * C1 + n2 * C2);

        // 编码并写入结果
        let value = createPayload(depth, n);
        storeResult(viewportSize, vec2u(u32(x), u32(y)), value);
      }

      // 水平步进: 沿 x 方向递加 A
      CX0 += CC0.A;
      CX1 += CC1.A;
      CX2 += CC2.A;
    }

    // 垂直步进: 换行时递加 B
    CY0 += CC0.B;
    CY1 += CC1.B;
    CY2 += CC2.B;
  }
}


///////////////////////////////////////////////////////////////
// Edge Function 及其增量形式
//
// edgeFunction(v0, v1, p) 计算:
//   (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x)
// 这等于有符号三角形面积的两倍。
//
// EdgeC: 将 edge function 改写为 A*x + B*y + C 的形式。
// 这样当逐像素遍历时:
//   - 向右移 1 像素: 只需 += A (因为 y 不变, C 不变)
//   - 向下移 1 像素: 只需 += B (因为 x 不变, C 不变)
//
// 避免每次重新计算完整的 edge function，~3x 加速。
// 参考: https://www.sctheblog.com/blog/hair-software-rasterize/#optimization-or-not
///////////////////////////////////////////////////////////////

/** 增量式 Edge Function 系数 */
struct EdgeC{ A: f32, B: f32, C: f32 }

/**
 * 从 edge function 提取 A, B, C 系数
 * edgeFunction(v0, v1, p) = (p.x - v0.x)*(v1.y - v0.y) - (p.y - v0.y)*(v1.x - v0.x)
 *                         = p.x*(v1.y - v0.y) + p.y*(-v1.x + v0.x) + (-v0.x*v1.y + v0.y*v1.x)
 *                         = A*p.x + B*p.y + C
 *   其中: A = v1.y - v0.y
 *         B = -v1.x + v0.x
 *         C = -v0.x*v1.y + v0.y*v1.x
 */
fn edgeC(v0: vec2f, v1: vec2f) -> EdgeC{
  var result: EdgeC;
  result.A = v1.y - v0.y;          // 水平步进增量
  result.B = -v1.x + v0.x;         // 垂直步进增量
  result.C = -v0.x * v1.y + v0.y * v1.x;  // 常数项
  return result;
}

/**
 * 标准 Edge Function (非增量式)
 * 返回有符号三角形面积 × 2
 * 正值 = 点在边内侧 (CCW)
 *
 * 参考: https://www.sctheblog.com/blog/hair-software-rasterize/#edge-function
 */
fn edgeFunction(v0: vec2f, v1: vec2f, p: vec2f) -> f32 {
  return (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
}


///////////////////////////////////////////////////////////////
// Payload 编码
//
// 由于 WGSL 没有 atomic<u64>, 必须把 depth 和 normal
// 压缩到单个 u32 中:
//
//   bits 16-31: depth (16 bits, 反转)
//   bits  8-15: normal.x (8 bits, octahedron 编码)
//   bits  0-7:  normal.y (8 bits, octahedron 编码)
//
// depth 反转 (1.0 - depth_ndc):
//   因为 atomicMax 只能取最大值, 反转後近的像素值更大。
//   WebGPU clear color 为 0, 所以 atomicMax 自动保留最近像素。
///////////////////////////////////////////////////////////////

const U16_MAX: f32 = 65535.0;

/** 将深度和法线编码为 u32 payload */
fn createPayload(depth0: f32, n: vec3f) -> u32 {
  // depth 反转: 近 = 大值, 配合 atomicMax
  let depth = 1.0 - depth0;
  // 量化到 16bit
  let depthU16 = clamp(depth * U16_MAX, 0., U16_MAX - 1);

  // 法线: octahedron 编码 → 8bit 量化
  let n_0_1 = encodeOctahedronNormal(n);  // [0, 1]
  let nPacked: u32 = (
    (u32(n_0_1.x * 255) << 8) |   // normal.x → bits 8-15
     u32(n_0_1.y * 255)            // normal.y → bits 0-7
  );

  // 组合: depth(高16位) | normal(低16位)
  return (u32(depthU16) << 16) | nPacked;
}


///////////////////////////////////////////////////////////////
// 存储结果到输出缓冲区
///////////////////////////////////////////////////////////////

/**
 * 将 payload 通过 atomicMax 写入像素缓冲区
 *
 * 使用 atomicMax 而非 atomicMin 的原因:
 *   - depth 反转(大的值=近)
 *   - WebGPU 清除到 0
 *   所以 atomicMax 自然实现了 "保留最近像素" 的深度测试
 *
 * NOTE: 如果要保存为 .png, 格式为 ABGR
 */
fn storeResult(viewportSize: vec2u, posPx: vec2u, value: u32) {
  // 边界检查
  if(
    posPx.x < 0 || posPx.x >= viewportSize.x ||
    posPx.y < 0 || posPx.y >= viewportSize.y
  ) {
    return;
  }
  // Y 轴翻转: WebGPU 坐标系 Y 向上, 图像 Y 向下
  let y = viewportSize.y - posPx.y - 1u;
  let idx: u32 = y * viewportSize.x + posPx.x;
  // 深度测试 (atomicMax = 保留最近像素)
  atomicMax(&_softwareRasterizerResult[idx], value);
}


///////////////////////////////////////////////////////////////
// 坐标变换工具
///////////////////////////////////////////////////////////////

/** 顶点 MVP 变换: 模型空间 → 裁剪空间 → NDC */
fn projectVertex(mvpMat:mat4x4f, pos: vec4f) -> vec3f {
  let posClip = mvpMat * pos;
  let posNDC = posClip / posClip.w;  // 透视除法
  return posNDC.xyz;
}

/** NDC [-1,1] → 屏幕像素坐标 [0, viewportSize] */
fn ndc2viewportPx(viewportSize: vec2f, pos: vec3f) -> vec2f {
  let pos_0_1 = pos.xy * 0.5 + 0.5;  // [-1,1] → [0,1]
  return pos_0_1 * viewportSize.xy;  // [0,1] → 像素坐标
}


///////////////////////////////////////////////////////////////
// 调试: 重心坐标可视化
///////////////////////////////////////////////////////////////

/** 将重心坐标编码为 RGBA 颜色, 用于调试 */
fn debugBarycentric(C0: f32, C1: f32, C2: f32) -> u32 {
  let color0: u32 = u32(C0 * 255);
  let color1: u32 = u32(C1 * 255);
  let color2: u32 = u32(C2 * 255);
  return (0xff000000u |       // 不透明
     color0 |                  // R = C0
    (color1 << 8) |            // G = C1
    (color2 << 16)             // B = C2
  );
}
