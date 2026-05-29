///////////////////////////////////////////////////////////////
// Billboard / Impostor Shader
//
// 渲染管线中的位置:
//   CullInstancesPass 中, 通过视锥+遮挡剔除的实例会分流:
//     - 屏幕面积 > billboardThreshold → 正常 meshlet 渲染 (CullMeshletsPass)
//     - 屏幕面积 ≤ billboardThreshold → Billboard (本 shader)
//
// Billboard 策略:
//   当一个实例在屏幕上投影面积很小时（远处/小物体），
//   不再用复杂的 meshlet LOD 系统渲染，而是用一个
//   始终面向相机的四边形 + 预渲染的 impostor 纹理来替代。
//   这样可以大幅减少远处物体的绘制开销。
//
// 核心机制:
//   1. 始终面对相机的四边形 (billboard quad)
//   2. 基于物体朝向角度, 从 12 张预渲染的 impostor 图片中采样
//   3. 相邻 impostor 之间做混合 (dither-based blending)
//   4. 完整 PBR 光照 (与硬件光栅化共享)
///////////////////////////////////////////////////////////////


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
     * 全局 Uniform 结构体 (与所有 Pass 共享)
     *
     * flags 位字段:
     *   bit 1        - meshlet 视锥剔除
     *   bit 2        - meshlet 遮挡剔除
     *   bits 3-5     - 着色模式 (1=三角形色, 2=meshlet色, 3=LOD色, 4=法线, 5=红色)
     *   bits 6-7     - 实例级剔除
     *   bits 8-11    - 调试深度金字塔 mipmap
     *   bits 12-15   - 调试遮挡剔除 mipmap 覆盖
     *   bit 16       - 强制 billboard
     *   bits 17-22   - billboard 抖动强度 (0-63 → [0,1])
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
    /// 强制所有实例使用 billboard (用于排查问题)
    fn useForceBillboards(flags: u32) -> bool { return checkFlag(flags, 65536u); }
    /// 着色模式 (bits 3-5)
    fn getShadingMode(flags: u32) -> u32 {
      return (flags >> 2u) & b111;
    }
    /// 调试深度金字塔层级 (bits 8-11)
    fn getDbgPyramidMipmapLevel(flags: u32) -> i32 {
      return i32(clamp((flags >> 8u) & b1111, 0u, 15u));
    }
    /// 遮挡剔除 mipmap 覆盖 (bits 12-15, 15=关闭)
    fn getOverrideOcclusionCullMipmap(flags: u32) -> i32 {
      let v: u32 = clamp((flags >> 12u) & b1111, 0u, 15u);
      if (v == 15u) { return -1; }
      return i32(v);
    }
    /// billboard 抖动强度 (bits 17-22, [0,63] → [0,1])
    fn getBillboardDitheringStrength(flags: u32) -> f32 {
      let v: u32 = (flags >> 17u) & b111111;
      return f32(v) / 63.0;
    }


///////////////////////////////////////////////////////////////
// 屏幕空间抖动 (Dithering)
//
// 用于 billboard 之间的平滑过渡。
// 使用 8x8 Bayer 抖动矩阵，在相邻 impostor 图像之间
// 做逐像素的抖动混合，避免突然切换造成的视觉突兀。
///////////////////////////////////////////////////////////////

/// Bayer 抖动矩阵的最大值
const DITHER_ELEMENT_RANGE: f32 = 63.0;
/// u8 颜色的最大级数 (256 级)
const DITHER_LINEAR_COLORSPACE_COLORS: f32 = 256.0;

/**
 * 8x8 Bayer 有序抖动矩阵 (Ordered Dithering)
 *
 * 每个元素的范围 [0, 63]。
 * 用于在相邻 impostor 之间做逐像素混合，
 * 避免视线变化时图像突然切换。
 *
 * 参考: https://en.wikipedia.org/wiki/Ordered_dithering
 *
 * 注: 不用纹理而用硬编码数组，因为纹理采样开销更大。
 */
const DITHER_MATRIX = array<u32, 64>(
  0, 32,  8, 40,  2, 34, 10, 42,
 48, 16, 56, 24, 50, 18, 58, 26,
 12, 44,  4, 36, 14, 46,  6, 38,
 60, 28, 52, 20, 62, 30, 54, 22,
  3, 35, 11, 43,  1, 33,  9, 41,
 51, 19, 59, 27, 49, 17, 57, 25,
 15, 47,  7, 39, 13, 45,  5, 37,
 63, 31, 55, 23, 61, 29, 53, 21
);

/**
 * 获取当前像素的抖动值 [0, 1]
 *
 * 基于屏幕坐标 (gl_FragCoord) 查 8x8 抖动矩阵，
 * 确保相邻像素有不同的抖动值，产生有序抖动效果。
 *
 * @param gl_FragCoord - 像素屏幕坐标
 */
fn getDitherForPixel(gl_FragCoord: vec2u) -> f32 {
  let pxPos = vec2u(
    gl_FragCoord.x % 8u,
    gl_FragCoord.y % 8u
  );
  let idx = pxPos.y * 8u + pxPos.x;
  // 注意: Naga 编译器不支持非常量索引 'array<u32, 64>'!
  // 已通过 'nagaFixes.ts' 解决
  let matValue = DITHER_MATRIX[idx]; // [1-64]
  return f32(matValue) / DITHER_ELEMENT_RANGE;
}

/**
 * 对颜色施加抖动调制
 * 用于相邻 impostor 之间的混合过渡
 */
fn ditherColor (
  gl_FragCoord: vec2u,
  originalColor: vec3f,
  strength: f32
) -> vec3f {
  let ditherMod = getDitherForPixel(gl_FragCoord) * strength / DITHER_LINEAR_COLORSPACE_COLORS;
  return originalColor + ditherMod;
}


///////////////////////////////////////////////////////////////
// 数据打包/解包
//
// Impostor 纹理的每个像素存储:
//   R: color (ABGR, pack4x8unorm)
//   G: normal (pack4x8snorm)
//   B: 未使用
//   A: 未使用
///////////////////////////////////////////////////////////////

/** 法线打包为 f32 (pack4x8snorm) */
fn packNormal(n: vec4f) -> f32 {
  return bitcast<f32>(pack4x8snorm(n));
}

/** 从 f32 解包法线 (unpack4x8snorm) */
fn unpackNormal(p: f32) -> vec3f {
  let v = unpack4x8snorm(bitcast<u32>(p));
  return normalize(v.xyz);
}

/** 颜色打包为 f32 (pack4x8unorm, ABGR) */
fn packColor8888(color: vec4f) -> f32 {
  return bitcast<f32>(pack4x8unorm(color));
}

/** 从 f32 解包颜色 (unpack4x8unorm, ABGR) */
fn unpackColor8888(p: f32) -> vec4f {
  return unpack4x8unorm(bitcast<u32>(p));
}


///////////////////////////////////////////////////////////////
// 法线 & 空间变换
///////////////////////////////////////////////////////////////

/**
 * 法线从模型空间→世界空间
 * WARNING: 仅当无缩放时正确
 */
fn transformNormalToWorldSpace(modelMat: mat4x4f, normalV: vec3f) -> vec3f {
  let normalMatrix = modelMat;
  let normalWS = normalMatrix * vec4f(normalV, 0.0);
  return normalize(normalWS.xyz);
}

/** Octahedron 法线编码 (3D→2D) */
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
// Disney PBR 光照模型 (与硬件光栅化完全一致)
//
// 包含: Lambert 漫反射 + Cook-Torrance 镜面反射
//   D: GGX 法线分布
//   G: Smith 自遮挡
//   F: Schlick 菲涅尔近似
///////////////////////////////////////////////////////////////

const DIELECTRIC_FRESNEL = vec3f(0.04, 0.04, 0.04);
const METALLIC_DIFFUSE_CONTRIBUTION = vec3(0.0, 0.0, 0.0);

fn pbr_LambertDiffuse(material: Material) -> vec3f {
  return material.albedo;
}

fn FresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn DistributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
    let a      = roughness * roughness;
    let a2     = a * a;
    let NdotH  = dotMax0(N, H);
    let NdotH2 = NdotH * NdotH;
    var denom = NdotH2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return a2 / denom;
}

fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    let denom = NdotV * (1.0 - k) + k;
    return NdotV / denom;
}

fn GeometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV = dotMax0(N, V);
    let NdotL = dotMax0(N, L);
    let ggx2  = GeometrySchlickGGX(NdotV, roughness);
    let ggx1  = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

fn pbr_CookTorrance(
  material: Material,
  V: vec3f,
  L: vec3f,
  F: ptr<function,vec3f>
) -> vec3f {
  let H = normalize(V + L);
  let N = material.normal;
  let F0 = mix(DIELECTRIC_FRESNEL, material.albedo, material.isMetallic);
  *F = FresnelSchlick(dotMax0(H, V), F0);
  let G = GeometrySmith(N, V, L, material.roughness);
  let NDF = DistributionGGX(N, H, material.roughness);
  let numerator = NDF * G * (*F);
  let denominator = 4.0 * dotMax0(N, V) * dotMax0(N, L);
  return numerator / max(denominator, 0.001);
}

fn pbr_mixDiffuseAndSpecular(material: Material, diffuse: vec3f, specular: vec3f, F: vec3f) -> vec3f {
  let kS = F;
  let kD = mix(vec3f(1.0, 1.0, 1.0) - kS, METALLIC_DIFFUSE_CONTRIBUTION, material.isMetallic);
  return kD * diffuse + specular;
}

fn disneyPBR(material: Material, light: Light) -> vec3f {
  let N = material.normal;
  let V = material.toEye;
  let L = normalize(light.position - material.positionWS);
  let attenuation = 1.0;
  let lambert = pbr_LambertDiffuse(material);
  var F: vec3f = vec3f();
  let specular = pbr_CookTorrance(material, V, L, &F);
  let NdotL = dotMax0(N, L);
  let brdfFinal = pbr_mixDiffuseAndSpecular(material, lambert, specular, F);
  let radiance = light.color * attenuation * light.intensity;
  return brdfFinal * radiance * NdotL;
}


///////////////////////////////////////////////////////////////
// 数据结构 & 常量
///////////////////////////////////////////////////////////////

const LIGHT_COUNT = 2u;
const PI: f32 = 3.141592653589793;

struct Material {
  positionWS: vec3f,
  normal: vec3f,
  toEye: vec3f,
  albedo: vec3f,
  roughness: f32,
  isMetallic: f32,
};

struct Light {
  position: vec3f,
  color: vec3f,
  intensity: f32
};

fn unpackLight(pos: vec3f, color: vec4f, light: ptr<function, Light>) {
  (*light).position = pos;
  (*light).color = color.rgb;
  (*light).intensity = color.a;
}

fn dotMax0 (n: vec3f, toEye: vec3f) -> f32 {
  return max(0.0, dot(n, toEye));
}

fn doShading(
  material: Material,
  ambientLight: vec4f,
  lights: array<Light, 2>
) -> vec3f {
  let ambient = ambientLight.rgb * ambientLight.a;
  var radianceSum = vec3(0.0);
  radianceSum += disneyPBR(material, lights[0]);
  radianceSum += disneyPBR(material, lights[1]);
  return ambient + radianceSum;
}


///////////////////////////////////////////////////////////////
// 场景灯光 & 默认材质
///////////////////////////////////////////////////////////////

const AMBIENT_LIGHT = vec4f(1., 1., 1., 0.05);
const LIGHT_FAR = 99999.0;

fn fillLightsData(lights: ptr<function, array<Light, LIGHT_COUNT>>){
  (*lights)[0].position = vec3f(LIGHT_FAR, LIGHT_FAR, 0);
  (*lights)[0].color = vec3f(1., 0.95, 0.8);
  (*lights)[0].intensity = 1.5;
  (*lights)[1].position = vec3f(-LIGHT_FAR, -LIGHT_FAR / 3.0, LIGHT_FAR / 3.0);
  (*lights)[1].color = vec3f(0.8, 0.8, 1.);
  (*lights)[1].intensity = 0.7;
}

/**
 * 创建默认材质 (与硬件光栅化一致)
 * - 浅灰反照率 (0.9)
 * - 高粗糙度 (0.8)
 * - 非金属
 * - 法线从屏幕导数重建
 */
fn createDefaultMaterial(
  material: ptr<function, Material>,
  positionWS: vec4f
){
  let cameraPos = _uniforms.cameraPosition.xyz;
  (*material).positionWS = positionWS.xyz;
  (*material).normal = normalFromDerivatives(positionWS);
  (*material).toEye = normalize(cameraPos - positionWS.xyz);
  (*material).albedo = vec3f(0.9, 0.9, 0.9);
  (*material).roughness = 0.8;
  (*material).isMetallic = 0.0;
}

/** 从屏幕导数重建法线 */
fn normalFromDerivatives(wsPosition: vec4f) -> vec3f{
  let posWsDx = dpdxFine(wsPosition);
  let posWsDy = dpdyFine(wsPosition);
  return normalize(cross(posWsDy.xyz, posWsDx.xyz));
}


///////////////////////////////////////////////////////////////
// GPU Buffer Bindings
///////////////////////////////////////////////////////////////

/**
 * Binding 2: 实例剔除结果 (CullParams)
 *
 * 由 CullInstancesPass 写入，包含:
 *   workgroupsX/Y/Z - 间接 dispatch 参数
 *   actuallyDrawnInstances - 可见实例数
 *   objectBoundingSphere - 物体包围球 (用于 billboard 大小)
 *   allMeshletsCount - meshlet 总数
 */
struct CullParams{
  workgroupsX: u32,
  workgroupsY: u32,
  workgroupsZ: u32,
  actuallyDrawnInstances: u32,
  objectBoundingSphere: vec4f,    // .xyz = 中心, .w = 半径
  allMeshletsCount: u32,
}
@group(0) @binding(2)
var<storage, read> _drawnInstancesParams: CullParams;


/// Binding 1: 实例变换矩阵数组
@group(0) @binding(1)
var<storage, read> _instanceTransforms: array<mat4x4<f32>>;

fn _getInstanceTransform(idx: u32) -> mat4x4<f32> {
  return _instanceTransforms[idx];
}

fn _getInstanceCount() -> u32 {
  return arrayLength(&_instanceTransforms);
}


/// Binding 3: Billboard 实例 ID 列表 (由 CullInstancesPass 写入)
/// 每个元素是需要渲染为 billboard 的实例在 instanceTransforms 中的索引
@group(0) @binding(3)
var<storage, read> _drawnImpostorsList: array<u32>;


/// Binding 4: Impostor 纹理
///
/// 特殊布局: 将 12 张 impostor 图片水平拼接在一张纹理中
///   [img0][img1][img2]...[img11]
/// 每张 impostor 存储:
///   R: 颜色 (ABGR, pack4x8unorm)
///   G: 法线 (pack4x8snorm)
///   B/A: 未使用
///
/// uv.x 映射到: (impostor_idx + uv.x) / IMPOSTOR_COUNT
@group(0) @binding(4)
var _diffuseTexture: texture_2d<f32>;

/// Binding 5: 纹理采样器
@group(0) @binding(5)
var _sampler: sampler;


///////////////////////////////////////////////////////////////
// 顶点着色器输出
///////////////////////////////////////////////////////////////

struct VertexOutput {
  @builtin(position) position: vec4<f32>,      // 裁剪空间位置
  @location(0) positionWS: vec4f,              // 世界空间位置 (用于光照)
  @location(1) uv: vec2f,                      // UV 坐标
  @location(2) @interpolate(flat) facingAngleDgr: f32,  // 朝向角度 (flat: 不插值)
  @location(3) @interpolate(flat) tfxIdx: u32,          // 实例变换索引
};


///////////////////////////////////////////////////////////////
// 顶点着色器 main_vs
//
// Billboarding 核心逻辑:
//   1. 在视图空间 (View Space) 创建一个始终面向相机的四边形
//   2. 四边形大小 = 包围球半径
//   3. 计算物体朝向角度 (忽略 Y 轴), 用于选择对应的 impostor 图像
//
// 原理:
//   普通的 billboard: 四边形始终正面朝向相机 (法线 = 视线方向)
//   本 shader 更进一步: 根据物体实际朝向选择预渲染的 impostor 图片,
//   看起来像是 3D 物体在转动, 而不是扁平的广告牌
///////////////////////////////////////////////////////////////

@vertex
fn main_vs(
  @builtin(vertex_index) inVertexIndex: u32,    // [0, 5], 6 个顶点组成四边形
  @builtin(instance_index) inInstanceIndex: u32, // billboard 实例索引
) -> VertexOutput {
  // 6 个顶点组成两个三角形 (覆盖一个四边形)
  // 不能移到函数外作为 const, Naga 编译限制
  var BILLBOARD_VERTICES = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),  // 左下
    vec2<f32>(-1.0, 1.0),   // 左上
    vec2<f32>(1.0, -1.0),   // 右下
    vec2<f32>(1.0, 1.0),    // 右上
    vec2<f32>(-1.0, 1.0),   // 左上 (三角形2)
    vec2<f32>(1.0, -1.0),   // 右下 (三角形2)
  );

  var result: VertexOutput;

  // 获取四边形偏移量 (NDC [-1,1])
  let quadOffset = BILLBOARD_VERTICES[inVertexIndex];
  // 获取 billboard 实例的变换索引
  let tfxIdx = _drawnImpostorsList[inInstanceIndex];
  let modelMat = _getInstanceTransform(tfxIdx);

  // 获取物体包围球, 用半径控制四边形大小
  let boundingSphere = _drawnInstancesParams.objectBoundingSphere;
  let r = boundingSphere.w;  // 包围球半径
  let viewMat = _uniforms.viewMatrix;
  let projMat = _uniforms.projMatrix;

  // Billboard 投影计算:
  //   1. 将包围球中心变换到视图空间 (view space)
  //   2. 在视图空间中, 用半径扩展四边形顶点:
  //        cornerVS = (center.xy + r * quadOffset, center.z, 1)
  //      由于在视图空间, XY 平面始终垂直于视线方向,
  //      所以四边形始终正面朝向相机
  //   3. 投影到裁剪空间
  //
  // 参考测试: './mathPlayground.test.ts'
  let center = viewMat * modelMat * vec4f(boundingSphere.xyz, 1.);
  let cornerVS = vec4f(center.xy + r * quadOffset, center.z, 1.);
  // TODO: 修改 .z 可控制近/远, 影响遮挡剔除的保守程度

  result.position = projMat * cornerVS;
  result.positionWS = modelMat * vec4f(boundingSphere.xyz, 1.);
  // UV: [-1,1] → [0,1], Y 翻转
  result.uv = (quadOffset.xy + 1.0) / 2.0;
  result.uv.y = 1.0 - result.uv.y;
  result.tfxIdx = tfxIdx;

  // 计算物体朝向相对于相机的 2D 角度 (忽略 Y 轴)
  // 用于从 12 张 impostor 图片中选择正确的角度
  let cameraPos = _uniforms.cameraPosition.xyz;
  let centerWS = (modelMat * vec4f(boundingSphere.xyz, 1.)).xyz;
  var camera2ModelDir: vec3f = cameraPos - centerWS;
  var objectFrontDir: vec3f = (modelMat * vec4f(0., 0., 1., 0.)).xyz;  // 物体的 Z 正向
  result.facingAngleDgr = angleDgr_axisXZ(objectFrontDir, camera2ModelDir);

  return result;
}


///////////////////////////////////////////////////////////////
// 角度计算工具
//
// 计算物体 Z 轴正方向与相机→物体方向在 XZ 平面上的夹角
// 用于选择正确的 impostor 图片角度
///////////////////////////////////////////////////////////////

/** 在 XZ 平面上计算两个向量的夹角 [0°, 360°)
 * https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
 *
 * 修改前请参考: 'src/passes/naniteBillboard/mathPlayground.test.ts'
 */
fn angleDgr_axisXZ(vecA: vec3f, vecB: vec3f) -> f32 {
  let vecAn = normalize(vec2f(vecA.x, vecA.z));
  let vecBn = normalize(vec2f(vecB.x, vecB.z));
  let dot = vecAn.x * vecBn.x + vecAn.y * vecBn.y;
  let det = vecAn.x * vecBn.y - vecAn.y * vecBn.x;
  var dgr = degrees(atan2(det, dot));  // [-180, 180]

  while (dgr < 0.0) { dgr += 360.0; } // 保证 [0, 360)
  return dgr;
}


///////////////////////////////////////////////////////////////
// Impostor 采样常量
///////////////////////////////////////////////////////////////

/// Impostor 图片数量 (12 张，每 30° 一张)
const IMPOSTOR_COUNT: u32 = 12;
const IMPOSTOR_COUNT_INV: f32 = 1.0 / f32(12);  // 1/12


/**
 * Impostor 采样结果
 */
struct ImpostorSample {
  diffuse: vec4f,   // 漫反射颜色 (含 alpha)
  normal: vec3f,    // 世界空间法线
};


///////////////////////////////////////////////////////////////
// 片段着色器 main_fs
//
// 核心:
//   1. 根据朝向角度, 从 12 张 impostor 中选择邻近的两张
//   2. 对这两张做混合 (dither-based blend, 避免突兀切换)
//   3. 对混合后的颜色和法线做 PBR 光照
//   4. alpha < 0.5 的像素丢弃
//
// 调试模式:
//   4 - 显示法线
//   5 - 显示蓝色
///////////////////////////////////////////////////////////////

@fragment
fn main_fs(
  fragIn: VertexOutput
) -> @location(0) vec4<f32> {
  let modelMat = _getInstanceTransform(fragIn.tfxIdx);
  let delta = 360.0 * IMPOSTOR_COUNT_INV;  // 30°
  let shownImageF32 = fragIn.facingAngleDgr / delta;  // 浮点角度索引

  // 选择相邻的两张 impostor
  let shownImage0 = u32(floor(shownImageF32));
  let shownImage1 = u32(ceil(shownImageF32));
  let impostor0 = impostorSample(modelMat, shownImage0, fragIn.uv);
  let impostor1 = impostorSample(modelMat, shownImage1, fragIn.uv);

  // 混合因子: dither-based blending
  // 用抖动 (dither) 代替简单的线性混合, 减少视觉伪影
  let ditherStr = getBillboardDitheringStrength(_uniforms.flags);
  let dither = getDitherForPixel(vec2u(fragIn.position.xy)) - 0.5;  // [-0.5, 0.5]
  let modStr = saturate(mix(fract(shownImageF32), dither, ditherStr));

  let shadingMode = getShadingMode(_uniforms.flags);
  var color: vec4f;

  if (shadingMode == 4u) {
    // 调试: 显示法线
    color = vec4f(abs(impostor0.normal), impostor0.diffuse.a);

  } else if (shadingMode == 5u) {
    // 调试: 纯蓝色
    color = vec4f(0., 0., 1., impostor0.diffuse.a);

  } else {
    // 默认: 完整 PBR 光照
    var material: Material;
    createDefaultMaterial(&material, fragIn.positionWS);
    var lights = array<Light, LIGHT_COUNT>();
    fillLightsData(&lights);

    // 对两张 impostor 分别做 PBR 光照, 再混合结果
    material.normal = impostor0.normal;
    material.albedo = impostor0.diffuse.rgb;
    let c0 = doShading(material, AMBIENT_LIGHT, lights);

    material.normal = impostor1.normal;
    material.albedo = impostor1.diffuse.rgb;
    let c1 = doShading(material, AMBIENT_LIGHT, lights);

    // 对光照结果做 dither-based 混合
    let a = mix(impostor0.diffuse.a, impostor1.diffuse.a, modStr);
    color = vec4f(mix(c0, c1, modStr), a);
  }

  // alpha 测试: 丢弃半透明像素
  if (color.a < 0.5) { discard; }
  return vec4(color.xyz, 1.0);
}


///////////////////////////////////////////////////////////////
// Impostor 纹理采样
//
// 纹理布局: 12 张图片水平拼接
//   uvX = (imageIdx + uv.x) / IMPOSTOR_COUNT
//
// 每个 texel 存储:
//   R: color (pack4x8unorm ABGR → f32)
//   G: normal (pack4x8snorm → f32)
//   B/A: 未使用
///////////////////////////////////////////////////////////////

fn impostorSample(modelMat: mat4x4f, idx: u32, uv: vec2f) -> ImpostorSample {
  // 水平偏移: 从 12 张图片中选择对应角度的图片
  // 例如: idx=4, uv.x=0.7 → uvX = (4+0.7)/12 = 0.392
  let uvX = (f32(idx % IMPOSTOR_COUNT) + uv.x) * IMPOSTOR_COUNT_INV;
  let texValues = textureSample(_diffuseTexture, _sampler, vec2f(uvX, uv.y));

  var result: ImpostorSample;
  // R = 颜色 (pack4x8unorm)
  result.diffuse = unpackColor8888(texValues.r);
  // G = 法线 (pack4x8snorm), 再变换到世界空间
  result.normal = transformNormalToWorldSpace(modelMat, unpackNormal(texValues.g));
  return result;
}
