///////////////////////////////////////////////////////////////
// 软硬光栅化合成 Shader (Rasterize Combine Pass)
//
// 渲染管线中的最后一步:
//   CullInstancesPass → CullMeshletsPass → 硬件光栅化 (color attachment)
//                                           └→ 软件光栅化 (atomic buffer)
//                                                    ↓
//                                         合成 Pass (本 shader)
//
// 为什么需要合成 Pass:
//   软件光栅化 (rasterizeSwPass) 将结果写入 atomic<u32> buffer,
//   而非 GPU 的 render target (color attachment)。
//   本 Pass 将 atomic buffer 中的 payload 解码,
//   以全屏三角形的片段着色器形式输出到最终的 color + depth attachment。
//
// 核心职责:
//   1. 读取软件光栅化 atomic buffer, 解码 (depth, normal)
//   2. 写入帧缓冲的 depth + color
//      - depth: 解码后的线性深度
//      - color: 用 decoded normal 做 PBR 光照
//   3. 如果像素在软件光栅化中没有被覆盖 → discard (保留硬件光栅化的结果)
//
// 深度测试协作:
//   - 硬件光栅化直接写入 render target (有深度测试)
//   - 软件光栅化写入 atomic buffer
//   - 合成 Pass 启用深度测试, 将软件结果叠加到硬件结果上
//   - 深度值写入正确, 下一帧的遮挡剔除基于最终深度
///////////////////////////////////////////////////////////////


/**
 * 生成全屏三角形的顶点位置 (无需顶点缓冲区)
 *
 * 原理: 通过 vertex_idx 的 XOR 运算生成覆盖全屏的三角形:
 *   vertIdx=0 → (-1,-1)  ← 此类技巧很常见
 *   vertIdx=1 → ( 3,-1)
 *   vertIdx=2 → (-1, 3)
 *
 * 参考:
 *   https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
 */
fn getFullscreenTrianglePosition(vertIdx: u32) -> vec4f {
  let outUV = vec2u((vertIdx << 1) & 2, vertIdx & 2);
  return vec4f(vec2f(outUV) * 2.0 - 1.0, 0.0, 1.0);
}


///////////////////////////////////////////////////////////////
// 法线 & 空间变换
///////////////////////////////////////////////////////////////

fn transformNormalToWorldSpace(modelMat: mat4x4f, normalV: vec3f) -> vec3f {
  let normalMatrix = modelMat;
  let normalWS = normalMatrix * vec4f(normalV, 0.0);
  return normalize(normalWS.xyz);
}

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

/** 从屏幕空间导数重建法线 */
fn normalFromDerivatives(wsPosition: vec4f) -> vec3f{
  let posWsDx = dpdxFine(wsPosition);
  let posWsDy = dpdyFine(wsPosition);
  return normalize(cross(posWsDy.xyz, posWsDx.xyz));
}


///////////////////////////////////////////////////////////////
// Disney PBR 光照 (与所有其他 Pass 共享)
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
// Uniforms & Flags
///////////////////////////////////////////////////////////////

    const b11 = 3u;
    const b111 = 7u;
    const b1111 = 15u;
    const b11111 = 31u;
    const b111111 = 63u;

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
// Buffer Bindings
///////////////////////////////////////////////////////////////

/**
 * Binding 1: 软件光栅化结果 buffer (只读)
 *
 * 由 rasterizeSwPass 写入, 每个像素一个 u32 payload:
 *   bits 16-31: depth (反转, 16bit, atomicMax)
 *   bits  8-15: normal.x (octahedron 编码, 8bit)
 *   bits  0-7:  normal.y (octahedron 编码, 8bit)
 */
@group(0) @binding(1)
var<storage, read> _softwareRasterizerResult: array<u32>;


///////////////////////////////////////////////////////////////
// 顶点着色器
//
// 无需顶点缓冲区, 直接用 built-in vertex_index 生成
// 全屏三角形的 3 个顶点位置
///////////////////////////////////////////////////////////////

@vertex
fn main_vs(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  return getFullscreenTrianglePosition(VertexIndex);
}


///////////////////////////////////////////////////////////////
// 片段着色器输出结构
///////////////////////////////////////////////////////////////

struct FragmentOutput {
  @builtin(frag_depth) fragDepth: f32,  // 写入深度缓冲区
  @location(0) color: vec4<f32>,        // 颜色输出
};


///////////////////////////////////////////////////////////////
// 片段着色器 main_fs (核心)
//
// 对软件光栅化的每个像素:
//   1. 从 atomic buffer 读取 u32 payload
//   2. 如果 payload == 0 → discard (该像素未被软件光栅化覆盖)
//   3. 解码 depth (反转 → NDC depth)
//   4. 解码法线 (octahedron → world normal)
//   5. 写入 depth: 软件光栅化的深度 (有深度测试, 覆盖硬件)
//   6. 写入 color: PBR 光照 (使用解码的法线)
//
// 调试模式:
//   4 - 显示法线
//   5 - 显示绿色
///////////////////////////////////////////////////////////////

@fragment
fn main_fs(
  @builtin(position) positionPxF32: vec4<f32>
) -> FragmentOutput {
  var result: FragmentOutput;

  let fragPositionPx = vec2u(positionPxF32.xy);

  // 1. 读取软件光栅化结果
  let viewportSize: vec2f = _uniforms.viewport.xy;
  // 计算一维索引 (row-major)
  let swRasterizerIdx: u32 = u32(positionPxF32.y) * u32(viewportSize.x) + u32(positionPxF32.x);
  let swRasterizerResU32: u32 = _softwareRasterizerResult[swRasterizerIdx];

  // 2. 跳过未被软件光栅化覆盖的像素
  if (swRasterizerResU32 == 0u){
    // 0 是 buffer clear 后的初始值
    // atomicMax 不可能写入 0, 因为:
    //   - depth 反转后最小也 > 0 (近平面)
    //   - octahedron 法线不可能全部为 0
    // 所以 0 代表"软件光栅化未覆盖此像素"
    discard;
  }

  // 3. 解码 depth
  //    高 16 位: depth (反转, 配合 atomicMax)
  let swRasterDepth: u32 = swRasterizerResU32 >> 16;
  //    反转回去: 1.0 - original_reversed_value
  //    得到 NDC depth (非线性的, 但符合 GPU 深度缓冲区预期)
  let swRasterDepthF32: f32 = 1.0 - f32(swRasterDepth) / 65535.0;

  // 写入深度! 这对于下一帧的遮挡剔除至关重要。
  // 如果不写深度, 遮挡剔除只会看到硬件光栅化的结果,
  // 而软件光栅化的小 meshlet 不会被任何物体遮挡。
  // 本 Pass 启用了深度测试 (写入 = 读取), 确保最终深度正确。
  result.fragDepth = swRasterDepthF32;

  // 4. 解码法线 (octahedron)
  //    bits 8-15: normal.x (8bit, [0,255] → octahedron [0,1])
  //    bits 0-7:  normal.y (8bit, [0,255] → octahedron [0,1])
  let nx = f32((swRasterizerResU32 >> 8) & 0xff) / 255.0;
  let ny = f32(swRasterizerResU32 & 0xff) / 255.0;
  let nUnpacked: vec3f = normalize(decodeOctahedronNormal(vec2f(nx, ny)));

  // 5. 着色
  let shadingMode = getShadingMode(_uniforms.flags);

  if (shadingMode == 4u) {
    // 调试: 显示法线
    result.color = vec4f(abs(nUnpacked.xyz), 1.0);

  } else if (shadingMode == 5u) {
    // 调试: 纯绿色 (标记软件光栅化像素)
    result.color = vec4f(0., 1., 0., 1.);

  } else {
    // 默认: 完整 PBR 光照
    var material: Material;

    // 用 VP 逆矩阵重建世界空间位置:
    //   position_proj = (pixelNDC, depthNDC, 1)
    //   position_ws = VP_inv * position_proj / w
    let positionProj = vec4(
      (positionPxF32.x / viewportSize.x) * 2.0 - 1.0,  // NDC x [-1,1]
      (positionPxF32.y / viewportSize.y) * 2.0 - 1.0,  // NDC y [-1,1]
      swRasterDepthF32,                                   // NDC depth
      1.0
    );
    var positionWs = _uniforms.vpMatrixInv * positionProj;
    positionWs = positionWs / positionWs.w; // 透视除法得到世界坐标

    createDefaultMaterial(&material, positionWs);
    material.normal = nUnpacked;
    material.roughness = 0.0;  // 软件光栅化设置为完全镜面反射

    // 光照
    var lights = array<Light, LIGHT_COUNT>();
    fillLightsData(&lights);
    result.color = vec4f(doShading(material, AMBIENT_LIGHT, lights), 1.0);
  }

  return result;
}
