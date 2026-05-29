///////////////////////////////////////////////////////////////
// 硬件光栅化 Shader (Hardware Rasterizer Shader)
//
// 渲染管线中的位置:
//   CullInstancesPass → CullMeshletsPass → 硬件光栅化(本shader)
//                                           └→ 软件光栅化
//
// CullMeshletsPass 将通过剔除的 meshlet 分流:
//   - 三角形数 ≤ softwareRasterizerThreshold → 软件光栅化
//   - 三角形数 >  softwareRasterizerThreshold → 硬件光栅化 (本 shader)
//
// 本 shader 包含:
//   1. 工具函数: MVP 矩阵、法线计算、PBR 光照
//   2. Uniforms & Flags 定义
//   3. Meshlet/Buffer Bindings
//   4. 顶点着色器 (main_vs): 读取 meshlet 数据，变换顶点
//   5. 片段着色器 (main_fs): PBR 光照 + 多种调试着色模式
///////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////
// 工具函数: MVP 矩阵 & 简易光照
///////////////////////////////////////////////////////////////

/**
 * 计算 Model-View-Projection 组合矩阵
 * MVP = Projection × View × Model
 * 将顶点从 模型空间 → 世界空间 → 视图空间 → 裁剪空间
 */
fn getMVP_Mat(modelMat: mat4x4<f32>, viewMat: mat4x4<f32>, projMat: mat4x4<f32>) -> mat4x4<f32> {
  let a = viewMat * modelMat;
  return projMat * a;
}


/**
 * 基于屏幕空间导数重建法线的简易光照
 * 仅用于快速可视化，不依赖顶点数据
 */
fn fakeLighting(wsPosition: vec4f) -> f32{
  let AMBIENT_LIGHT = 0.1;
  let LIGHT_DIR = vec3(5., 5., 5.);

  let normal = normalFromDerivatives(wsPosition);
  let lightDir = normalize(LIGHT_DIR);
  let NdotL = max(0.0, dot(normal.xyz, lightDir));
  return mix(AMBIENT_LIGHT, 1.0, NdotL);
}


///////////////////////////////////////////////////////////////
// 调试用调色板
///////////////////////////////////////////////////////////////

/// 14 种预设颜色，用于调试可视化
const COLOR_COUNT = 14u;
const COLORS = array<vec3f, COLOR_COUNT>(
    vec3f(1., 1., 1.),   // 白色
    vec3f(1., 0., 0.),   // 红色
    vec3f(0., 1., 0.),   // 绿色
    vec3f(0., 0., 1.),   // 蓝色
    vec3f(1., 1., 0.),   // 黄色
    vec3f(0., 1., 1.),   // 青色
    vec3f(1., 0., 1.),   // 品红

    vec3f(.5, .5, .5),   // 灰色
    vec3f(.5, 0., 0.),   // 暗红
    vec3f(.5, .5, 0.),   // 暗黄
    vec3f(0., 0., .5),   // 暗蓝
    vec3f(.5, .5, 0.),   // 暗黄 (重复)
    vec3f(0., .5, .5),   // 暗青
    vec3f(.5, 0., .5),   // 暗品红
);

/// 通过索引获取调色板颜色 (循环取模)
fn getRandomColor(idx: u32) -> vec3f {
  let color: vec3f = COLORS[idx % COLOR_COUNT];
  return color;
}


///////////////////////////////////////////////////////////////
// 法线计算
///////////////////////////////////////////////////////////////

/**
 * 从屏幕空间偏导数重建法线 (面法线)
 *
 * 原理: dpdxFine/dpdyFine 计算当前片段在屏幕上的梯度，
 * 两个梯度向量的叉积得到面法线
 * 优点: 不需要顶点法线属性
 * 缺点: 只能得到面法线，无法平滑着色
 */
fn normalFromDerivatives(wsPosition: vec4f) -> vec3f{
  let posWsDx = dpdxFine(wsPosition);  // 世界坐标对屏幕 X 的偏导
  let posWsDy = dpdyFine(wsPosition);  // 世界坐标对屏幕 Y 的偏导
  return normalize(cross(posWsDy.xyz, posWsDx.xyz));
}


/**
 * 将法线从模型空间变换到世界空间
 *
 * WARNING: 仅当模型矩阵无缩放 (仅旋转+平移) 时正确！
 * 有非均匀缩放时应使用 法线矩阵 = transpose(inverse(modelMat))
 *
 * 参考: https://paroj.github.io/gltut/Illumination/Tut09%20Normal%20Transformation.html
 */
fn transformNormalToWorldSpace(modelMat: mat4x4f, normalV: vec3f) -> vec3f {
  // 仅旋转时可直接用 modelMat 作为法线矩阵, w=0 忽略平移
  let normalMatrix = modelMat;
  let normalWS = normalMatrix * vec4f(normalV, 0.0);
  return normalize(normalWS.xyz);
}


///////////////////////////////////////////////////////////////
// Octahedron 法线编码/解码
//
// 将 3D 单位向量压缩为 2D (2个float)，减少存储开销
// 论文: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
// 原理: 将单位球面投影到正八面体 → 展开为 2D 正方形
///////////////////////////////////////////////////////////////

/** 八面体编码的折叠函数：将下半球的点翻折到上半平面 */
fn OctWrap(v: vec2f) -> vec2f {
  // select(f, t, cond): cond=true 返回 t, 否则返回 f
  let signX = select(-1.0, 1.0, v.x >= 0.0);
  let signY = select(-1.0, 1.0, v.y >= 0.0);
  return (1.0 - abs(v.yx)) * vec2f(signX, signY);
}

/**
 * 编码: 单位法线 → vec2 (范围 [0,1])
 *
 * 步骤:
 *   1. 法线投射到 L1 球面 (|x|+|y|+|z|=1 的八面体)
 *   2. 若 z<0 (下半球), OctWrap 折叠到上平面
 *   3. 结果从 [-1,1] 映射到 [0,1]
 */
fn encodeOctahedronNormal(n0: vec3f) -> vec2f {
  var n = n0 / (abs(n0.x) + abs(n0.y) + abs(n0.z));
  if (n.z < 0.0) {
    let a = OctWrap(n.xy);
    n.x = a.x;
    n.y = a.y;
  }
  return n.xy * 0.5 + 0.5;
}

/**
 * 解码: vec2 (范围 [0,1]) → 单位法线
 * 编码的逆过程
 */
fn decodeOctahedronNormal(f_: vec2f) -> vec3f {
  let f = f_ * 2.0 - 1.0;

  // 从 x,y 重建 z（满足 L1 范数约束 |x|+|y|+|z|=1）
  var n = vec3f(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
  // 若 z<0, 反折叠
  let t = saturate(-n.z);
  if (n.x >= 0.0){ n.x -= t; } else { n.x += t; }
  if (n.y >= 0.0){ n.y -= t; } else { n.y += t; }
  return normalize(n);
}


///////////////////////////////////////////////////////////////
// PBR (Physically Based Rendering) 光照模型
//
// Cook-Torrance BRDF = D(GGX) * G(Smith) * F(Schlick)
// 参考: 'Real Shading in Unreal Engine 4' by Brian Karis
///////////////////////////////////////////////////////////////

/// 电介体菲涅尔基础反射率 (~4%)
const DIELECTRIC_FRESNEL = vec3f(0.04, 0.04, 0.04);
/// 金属不贡献漫反射
const METALLIC_DIFFUSE_CONTRIBUTION = vec3(0.0, 0.0, 0.0);


/**
 * Lambert 漫反射
 * 物理上应除以 PI, 这里省略以降低计算量
 */
fn pbr_LambertDiffuse(material: Material) -> vec3f {
  return material.albedo;
}


/**
 * F - Fresnel 项 (Schlick 近似)
 *
 * 描述: 视角越接近掠射角，反射越强
 * F(θ) = F0 + (1-F0) * (1-cosθ)^5
 *
 * @param cosTheta - cos(视线方向 V, 半程向量 H)
 * @param F0 - 0° 入射的基础反射率
 *              非金属: ~0.04
 *              金属: = albedo
 */
fn FresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

/**
 * D - 法线分布函数 (GGX / Trowbridge-Reitz)
 *
 * 描述: 微表面法线与半程向量对齐的比例
 * D(θ) = α² / (π * (cos²θ * (α²-1) + 1)²)
 * 其中 α = roughness²
 * 粗糙度低 → 高光集中; 粗糙度高 → 高光扩散
 *
 * 参考: UE4 方程 2
 *
 * @param N - 法线
 * @param H - 半程向量 (V+L 归一化)
 * @param roughness - 粗糙度 [0,1]
 */
fn DistributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
    let a      = roughness * roughness;
    let a2     = a * a;             // α² = roughness⁴
    let NdotH  = dotMax0(N, H);
    let NdotH2 = NdotH * NdotH;

    var denom = NdotH2 * (a2 - 1.0) + 1.0;
    denom = PI * denom * denom;
    return a2 / denom;
}

/**
 * G - Smith 自遮挡函数 (Schlick-GGX 近似)
 *
 * 计算单一方向 (视线或光线) 的几何遮挡
 * G(θ) = NdotV / (NdotV * (1-k) + k)
 * 其中 k = (roughness+1)² / 8
 *
 * 参考: UE4 方程 4 第 1,2 行
 */
fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;  // IBL 版本的 k
    let denom = NdotV * (1.0 - k) + k;
    return NdotV / denom;
}

/**
 * G - Smith 联合遮挡函数
 *
 * G = G_light * G_view
 * 同时对光线方向和视线方向做遮挡检查
 *
 * 参考: UE4 方程 4 第 3 行
 */
fn GeometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV = dotMax0(N, V);
    let NdotL = dotMax0(N, L);
    let ggx2  = GeometrySchlickGGX(NdotV, roughness);  // 视线方向
    let ggx1  = GeometrySchlickGGX(NdotL, roughness);  // 光线方向
    return ggx1 * ggx2;
}

/**
 * Cook-Torrance 镜面反射 BRDF
 *
 * f_specular = D * G * F / (4 * NdotV * NdotL)
 *
 * @param material - 材质参数
 * @param V - 视线方向 (片段→相机)
 * @param L - 光线方向 (片段→光源)
 * @param F - 输出菲涅尔项 (用于外部混合漫反射/镜面反射)
 */
fn pbr_CookTorrance(
  material: Material,
  V: vec3f,
  L: vec3f,
  F: ptr<function,vec3f>  // 输出参数: ptr 引用传递
) -> vec3f {
  let H = normalize(V + L); // 半程向量
  let N = material.normal;  // 法线

  // F - Fresnel 项: 金属的 F0 = albedo
  let F0 = mix(DIELECTRIC_FRESNEL, material.albedo, material.isMetallic);
  *F = FresnelSchlick(dotMax0(H, V), F0);
  // G - 自遮挡
  let G = GeometrySmith(N, V, L, material.roughness);
  // D - 法线分布
  let NDF = DistributionGGX(N, H, material.roughness);

  // Cook-Torrance
  let numerator = NDF * G * (*F);
  let denominator = 4.0 * dotMax0(N, V) * dotMax0(N, L);
  return numerator / max(denominator, 0.001); // 防除零
}

/**
 * 混合漫反射和镜面反射 (能量守恒)
 *
 * 金属: kD ≈ 0, 颜色完全来自镜面反射
 * 非金属: kD = 1 - kS, 漫反射+镜面反射
 */
fn pbr_mixDiffuseAndSpecular(material: Material, diffuse: vec3f, specular: vec3f, F: vec3f) -> vec3f {
  let kS = F;  // 镜面反射比例 = Fresnel
  // 金属几乎无漫反射
  let kD = mix(vec3f(1.0, 1.0, 1.0) - kS, METALLIC_DIFFUSE_CONTRIBUTION, material.isMetallic);
  return kD * diffuse + specular;
}


/**
 * Disney PBR 完整光照计算 (单个光源)
 *
 * 最终光照 = (kD * diffuse + specular) * 入射辐射度 * NdotL
 */
fn disneyPBR(material: Material, light: Light) -> vec3f {
  let N = material.normal;                      // 法线
  let V = material.toEye;                       // 视线方向
  let L = normalize(light.position - material.positionWS);  // 光线方向
  let attenuation = 1.0;  // 硬编码无衰减 (演示用)

  // 漫反射
  let lambert = pbr_LambertDiffuse(material);

  // 镜面反射 (Cook-Torrance)
  var F: vec3f = vec3f();
  let specular = pbr_CookTorrance(material, V, L, &F);

  // 混合 + 光照入射
  let NdotL = dotMax0(N, L);
  let brdfFinal = pbr_mixDiffuseAndSpecular(material, lambert, specular, F);
  let radiance = light.color * attenuation * light.intensity;
  return brdfFinal * radiance * NdotL;
}


///////////////////////////////////////////////////////////////
// 常量 & 数据结构
///////////////////////////////////////////////////////////////

const LIGHT_COUNT = 2u;   // 固定 2 盏灯 (WGSL 不支持动态长度数组)
const PI: f32 = 3.141592653589793;


/**
 * PBR 材质结构体
 */
struct Material {
  positionWS: vec3f,         // 世界空间位置 (用于光照计算)
  normal: vec3f,             // 世界空间法线
  toEye: vec3f,              // 视线方向 (片段→相机)
  // Disney PBR 参数:
  albedo: vec3f,             // 基础颜色 (RGB)
  roughness: f32,            // 粗糙度 [0,1]: 0=镜面, 1=粗糙
  isMetallic: f32,           // 金属度 [0,1]: 0=非金属, 1=金属
};

/**
 * 光源结构体
 */
struct Light {
  position: vec3f,           // 世界空间位置
  color: vec3f,              // 颜色 (RGB)
  intensity: f32             // 强度
};

/** 从 uniform 解包光源 */
fn unpackLight(pos: vec3f, color: vec4f, light: ptr<function, Light>) {
  (*light).position = pos;
  (*light).color = color.rgb;
  (*light).intensity = color.a;
}

/** 安全的点积 (负值钳位到 0) */
fn dotMax0 (n: vec3f, toEye: vec3f) -> f32 {
  return max(0.0, dot(n, toEye));
}

/**
 * 完整着色: 环境光 + N 盏灯 PBR 累加
 *
 * 注: 因 Naga 编译器限制不能非常量索引数组，
 * 灯数固定为 2 且手动展开
 */
fn doShading(
  material: Material,
  ambientLight: vec4f,
  lights: array<Light, 2>
) -> vec3f {
  let ambient = ambientLight.rgb * ambientLight.a;
  var radianceSum = vec3(0.0);

  // Naga 编译限制: 不能 for 循环非常量索引
  radianceSum += disneyPBR(material, lights[0]);
  radianceSum += disneyPBR(material, lights[1]);

  return ambient + radianceSum;
}


///////////////////////////////////////////////////////////////
// 场景灯光 & 默认材质配置
///////////////////////////////////////////////////////////////

/// 微弱白色环境光
const AMBIENT_LIGHT = vec4f(1., 1., 1., 0.05);
/// 灯的位置放极远处 → 模拟方向光
const LIGHT_FAR = 99999.0;

/**
 * 填充两盏灯:
 *   灯0: 暖色主光 (右上)
 *   灯1: 冷色补光 (左下)
 */
fn fillLightsData(
  lights: ptr<function, array<Light, LIGHT_COUNT>>
){
  (*lights)[0].position = vec3f(LIGHT_FAR, LIGHT_FAR, 0);
  (*lights)[0].color = vec3f(1., 0.95, 0.8);    // 暖白
  (*lights)[0].intensity = 1.5;

  (*lights)[1].position = vec3f(-LIGHT_FAR, -LIGHT_FAR / 3.0, LIGHT_FAR / 3.0);
  (*lights)[1].color = vec3f(0.8, 0.8, 1.);     // 冷白
  (*lights)[1].intensity = 0.7;
}

/**
 * 创建默认材质:
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


///////////////////////////////////////////////////////////////
// Uniforms & Flags 定义
// (与所有 Pass 共享同一布局)
///////////////////////////////////////////////////////////////

    /// 位掩码常量
    const b11 = 3u;       // 2 bits 掩码
    const b111 = 7u;      // 3 bits 掩码
    const b1111 = 15u;    // 4 bits 掩码
    const b11111 = 31u;   // 5 bits 掩码
    const b111111 = 63u;  // 6 bits 掩码

    /**
     * 全局 Uniform 结构体
     *
     * flags 位字段布局:
     *   bit  1         - meshlet 视锥剔除开关
     *   bit  2         - meshlet 遮挡剔除开关
     *   bits 3-5       - 着色模式 (1=三角形色, 2=meshlet色, 3=LOD色, 4=法线, 5=红色)
     *   bits 6-7       - 实例级剔除开关
     *   bits 8-11      - 调试: 渲染深度金字塔 mipmap 层级 (0-15)
     *   bits 12-15     - 调试: 覆盖遮挡剔除 mipmap 层级 (0-14 强制, 15=关闭)
     *   bit  16        - 强制 billboard 模式
     *   bits 17-22     - billboard 抖动强度 (0-63 → [0,1])
     *   bits 23-32     - 未使用
     */
    struct Uniforms {
      vpMatrix: mat4x4<f32>,                    // VP 组合矩阵
      vpMatrixInv: mat4x4<f32>,                 // VP 逆矩阵
      viewMatrix: mat4x4<f32>,                  // View 矩阵
      projMatrix: mat4x4<f32>,                  // Projection 矩阵
      viewport: vec4f,                          // 视口尺寸
      cameraPosition: vec4f,                    // 相机位置
      cameraFrustumPlane0: vec4f,               // 视锥平面 0
      cameraFrustumPlane1: vec4f,               // 视锥平面 1
      cameraFrustumPlane2: vec4f,               // 视锥平面 2
      cameraFrustumPlane3: vec4f,               // 视锥平面 3
      cameraFrustumPlane4: vec4f,               // 视锥平面 4
      cameraFrustumPlane5: vec4f,               // 视锥平面 5
      flags: u32,                                // 标志位
      billboardThreshold: f32,                   // Billboard 屏幕面积阈值
      softwareRasterizerThreshold: f32,          // 软件光栅化阈值
      padding0: u32,                             // 对齐填充
      colorMgmt: vec4f,                          // 色彩管理
    };

    @binding(0) @group(0)
    var<uniform> _uniforms: Uniforms;

    /// 检查 flags 特定位
    fn checkFlag(flags: u32, bit: u32) -> bool { return (flags & bit) > 0; }
    fn useFrustumCulling(flags: u32) -> bool { return checkFlag(flags, 1u); }
    fn useOcclusionCulling(flags: u32) -> bool { return checkFlag(flags, 2u); }
    fn useInstancesFrustumCulling(flags: u32) -> bool { return checkFlag(flags, 32u); }
    fn useInstancesOcclusionCulling(flags: u32) -> bool { return checkFlag(flags, 64u); }
    fn useForceBillboards(flags: u32) -> bool { return checkFlag(flags, 65536u); }
    /// 获取着色模式 (bits 3-5)
    fn getShadingMode(flags: u32) -> u32 {
      return (flags >> 2u) & b111;
    }
    /// 获取调试深度金字塔 mipmap 层级 (bits 8-11)
    fn getDbgPyramidMipmapLevel(flags: u32) -> i32 {
      return i32(clamp((flags >> 8u) & b1111, 0u, 15u));
    }
    /// 获取遮挡剔除 mipmap 覆盖 (bits 12-15, 15=关闭)
    fn getOverrideOcclusionCullMipmap(flags: u32) -> i32 {
      let v: u32 = clamp((flags >> 12u) & b1111, 0u, 15u);
      if (v == 15u) { return -1; }
      return i32(v);
    }
    /// 获取 billboard 抖动强度 (bits 17-22, 0-63 → [0,1])
    fn getBillboardDitheringStrength(flags: u32) -> f32 {
      let v: u32 = (flags >> 17u) & b111111;
      return f32(v) / 63.0;
    }


///////////////////////////////////////////////////////////////
// Meshlet 数据结构 & GPU Buffer Bindings
///////////////////////////////////////////////////////////////

/**
 * Nanite Meshlet 树节点
 *
 * 组织为 LOD 树: 父节点 = 多个子节点的简化合并
 * LOD 判定需要同时比较当前误差和父节点误差
 */
struct NaniteMeshletTreeNode {
  boundsMidPointAndError: vec4f,       // .xyz = 兄弟合并包围球中心, .w = 当前节点简化误差 (clusterError)
  parentBoundsMidPointAndError: vec4f, // .xyz = 父节点包围球中心, .w = 父节点简化误差 (parentError)
  ownBoundingSphere: vec4f,            // .xyz = 自身包围球中心, .w = 自身包围球半径
  triangleCount: u32,                  // 三角形数量
  firstIndexOffset: u32,               // 在索引缓冲区中的偏移
  lodLevel: u32,                       // LOD 层级 (用于调试着色)
  padding0: u32,                       // 对齐填充 (凑齐 vec4 对齐)
}

/// Binding 1: Meshlet 树节点数组 (只读)
@group(0) @binding(1)
var<storage, read> _meshlets: array<NaniteMeshletTreeNode>;


/**
 * Binding 2: 通过剔除的 meshlet 绘制列表
 *
 * 每个元素是 vec2u:
 *   .x = instanceIdx (实例变换索引)
 *   .y = meshletIdx  (meshlet 索引)
 *
 * 存储策略 (由 CullMeshletsPass 写入):
 *   - 硬件光栅化 meshlet: 从头部向前写入 (正序读取)
 *   - 软件光栅化 meshlet: 从尾部向后写入 (倒序读取)
 * 双端写入避免冲突，共享同一 buffer
 */
@group(0) @binding(2)
var<storage, read> _drawnMeshletsList: array<vec2<u32>>;

/// 从列表头部读取硬件光栅化 meshlet 数据
fn _getMeshletHardwareDraw(idx: u32) -> vec2u {
  return _drawnMeshletsList[idx];
}

/// 从列表尾部倒序读取软件光栅化 meshlet 数据
fn _getMeshletSoftwareDraw(idx: u32) -> vec2u {
  let len: u32 = arrayLength(&_drawnMeshletsList);
  let idx2: u32 = len - 1u - idx; // 软件数据存储在尾部
  return _drawnMeshletsList[idx2];
}


/**
 * Binding 4: 顶点位置缓冲区
 *
 * WARNING: WGSL 不支持 'array<vec3f>' 作为 SSBO!
 * 必须用 'array<vec4f>'，否则运行时出错。
 * 花了大量时间调试才发现的坑。
 */
@group(0) @binding(4)
var<storage, read> _vertexPositionsNative: array<vec4f>;

fn _getVertexPosition(idx: u32) -> vec4f { return _vertexPositionsNative[idx]; }


/**
 * Binding 5: 顶点法线缓冲区 (octahedron 压缩编码)
 * 每法线仅 2 个 float (vec2f)，节省一半显存
 */
@group(0) @binding(5)
var<storage, read> _vertexNormals: array<vec2f>;

fn _getVertexNormal(idx: u32) -> vec3f {
  return decodeOctahedronNormal(_vertexNormals[idx]);
}


/// Binding 6: 顶点 UV 坐标
@group(0) @binding(6)
var<storage, read> _vertexUV: array<vec2f>;

fn _getVertexUV(idx: u32) -> vec2f { return _vertexUV[idx]; }


/**
 * Binding 3: 实例变换矩阵
 * 每个实例一个 mat4x4<f32> 模型矩阵
 */
@group(0) @binding(3)
var<storage, read> _instanceTransforms: array<mat4x4<f32>>;

fn _getInstanceTransform(idx: u32) -> mat4x4<f32> {
  return _instanceTransforms[idx];
}

fn _getInstanceCount() -> u32 {
  return arrayLength(&_instanceTransforms);
}


/// Binding 7: 索引缓冲区 (meshlet 三角形索引)
@group(0) @binding(7)
var<storage, read> _indexBuffer: array<u32>;


/// Binding 8: 漫反射纹理
@group(0) @binding(8)
var _diffuseTexture: texture_2d<f32>;

/// Binding 9: 纹理采样器
@group(0) @binding(9)
var _sampler: sampler;


///////////////////////////////////////////////////////////////
// 顶点着色器输出
///////////////////////////////////////////////////////////////

struct VertexOutput {
  @builtin(position) position: vec4<f32>,     // 裁剪空间位置 (GPU 自动处理)
  @location(0) positionWS: vec4f,             // 世界空间位置
  @location(1) normalWS: vec3f,               // 世界空间法线
  @location(2) uv: vec2f,                     // 纹理坐标
  @location(3) @interpolate(flat) instanceIdx: u32,   // 实例索引 (flat: 不插值)
  @location(4) @interpolate(flat) meshletId: u32,     // meshlet 索引 (flat)
  @location(5) @interpolate(flat) triangleIdx: u32,   // 三角形索引 (flat, 调试用)
};

/// 顶点被裁剪时的哨兵坐标 (移到极远处)
const OUT_OF_SIGHT = 9999999.0;


///////////////////////////////////////////////////////////////
// 顶点着色器 main_vs
//
// 输入:
//   vertex_index:   当前 meshlet 实例内的顶点索引 [0, triangleCount*3)
//   instance_index: 实例索引 (映射到 _drawnMeshletsList)
//
// 要点:
//   - 每个硬件光栅化的 (meshlet, 实例) 被绘制为 GPU 实例
//   - 每个实例固定绘制 MAX_MESHLET_TRIANGLES*3 个顶点
//   - 超出 meshlet.triangleCount*3 的顶点被推到屏幕外丢弃
//     这种策略节省了存储 (只需在列表存 vec2u, 不存顶点范围)
///////////////////////////////////////////////////////////////

@vertex
fn main_vs(
  @builtin(vertex_index) inVertexIndex: u32,
  @builtin(instance_index) inInstanceIndex: u32,
) -> VertexOutput {
  var result: VertexOutput;

  // 从绘制列表读取: 获取实例变换索引 和 meshlet 索引
  let drawData: vec2u = _getMeshletHardwareDraw(inInstanceIndex);
  // drawData.x = 实例变换索引, drawData.y = meshlet 索引

  let meshlet = _meshlets[drawData.y];
  result.meshletId = drawData.y;
  let modelMat = _getInstanceTransform(drawData.x);

  // 策略: 始终绘制 MAX_MESHLET_TRIANGLES*3 个顶点
  // 短 meshlet (少于最大三角形) 的额外顶点丢弃
  // 优点: 只存 vec2u/实例, 不存每个 meshlet 的顶点范围
  if (inVertexIndex >= meshlet.triangleCount * 3) {
    // 将超出三角形移到视口外:
    // WGSL 没有 NaN, 用较大的坐标值让 GPU 裁剪
    result.position.x = OUT_OF_SIGHT;
    result.position.y = OUT_OF_SIGHT;
    result.position.z = OUT_OF_SIGHT;
    result.position.w = 1.0;
    return result;
  }

  // 读取三角形顶点: 索引缓冲区 → 位置/法线/UV
  let vertexIdx = _indexBuffer[meshlet.firstIndexOffset + inVertexIndex];
  let vertexPos = _getVertexPosition(vertexIdx);
  let vertexN = _getVertexNormal(vertexIdx);
  let vertexUV = _getVertexUV(vertexIdx);

  // MVP 变换: 模型空间 → 裁剪空间
  let mvpMatrix = getMVP_Mat(modelMat, _uniforms.viewMatrix, _uniforms.projMatrix);
  let projectedPosition = mvpMatrix * vertexPos;
  let positionWS = modelMat * vertexPos;

  // 填充输出
  result.position = projectedPosition;
  result.positionWS = positionWS;
  result.normalWS = transformNormalToWorldSpace(modelMat, vertexN);
  result.uv = vertexUV;
  result.instanceIdx = drawData.x;
  result.triangleIdx = inVertexIndex;

  return result;
}


///////////////////////////////////////////////////////////////
// 片段着色器 main_fs
//
// 支持多种着色模式 (通过 flags 控制):
//   0 - 默认 PBR (纹理 + 光照)
//   1 - 按三角形随机着色
//   2 - 按 meshlet 随机着色
//   3 - 按 LOD 层级随机着色
//   4 - 法线可视化 (绝对值)
//   5 - 纯红色
///////////////////////////////////////////////////////////////

@fragment
fn main_fs(fragIn: VertexOutput) -> @location(0) vec4<f32> {
  let shadingMode = getShadingMode(_uniforms.flags);
  var color: vec3f;

  // 调试模式 1: 按三角形随机着色
  if (shadingMode == 1u) {
    color = getRandomColor(fragIn.triangleIdx);

  // 调试模式 2: 按 meshlet 随机着色
  } else if (shadingMode == 2u) {
    color = getRandomColor(fragIn.meshletId);

  // 调试模式 3: 按 LOD 层级随机着色
  // 不同颜色表示当前片段属于哪个 LOD 级别
  } else if (shadingMode == 3u) {
    let meshlet = _meshlets[fragIn.meshletId];
    let lodLevel = meshlet.lodLevel;
    color = getRandomColor(lodLevel);

  // 调试模式 4: 法线可视化
  } else if (shadingMode == 4u) {
    color = abs(normalize(fragIn.normalWS));

  // 调试模式 5: 纯红色
  } else if (shadingMode == 5u) {
    color = vec3f(1., 0., 0.);

  // 默认模式 0: 完整 PBR 光照
  } else {
    var material: Material;
    createDefaultMaterial(&material, fragIn.positionWS);
    material.normal = normalize(fragIn.normalWS);
    // 从纹理采样反照率
    material.albedo = textureSample(_diffuseTexture, _sampler, fragIn.uv).rgb;

    // 光照计算: 环境光 + 2 盏灯 PBR
    var lights = array<Light, LIGHT_COUNT>();
    fillLightsData(&lights);
    color = doShading(material, AMBIENT_LIGHT, lights);
  }

  return vec4(color.xyz, 1.0);
}
