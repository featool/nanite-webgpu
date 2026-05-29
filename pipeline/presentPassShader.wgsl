///////////////////////////////////////////////////////////////
// 呈现 Pass Shader (Present / Tonemapping Pass)
//
// 渲染管线中的最后一步 (Post-processing):
//   深度金字塔 → 渲染/合成 → Billboard → 呈现 Pass (本 shader)
//
// 核心职责:
//   将 HDR 渲染结果转换为 SDR 屏幕输出:
//     1. 屏幕抖动 (Dithering) - 减少颜色条带伪影
//     2. 曝光调整 (Exposure)
//     3. ACES 色调映射 (Tonemapping) - 将 HDR 压缩到 SDR
//     4. Gamma 校正 (Gamma Correction)
//
// 这是纯粹的 2D 后处理, 无需 3D 场景数据,
// 只需从渲染好的 color attachment 中读取 texel 并处理。
///////////////////////////////////////////////////////////////


/**
 * 生成全屏三角形的顶点位置
 * 无需顶点缓冲区, 通过 vertex_index 的位运算构造
 *
 * 参考: https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
 */
fn getFullscreenTrianglePosition(vertIdx: u32) -> vec4f {
  let outUV = vec2u((vertIdx << 1) & 2, vertIdx & 2);
  return vec4f(vec2f(outUV) * 2.0 - 1.0, 0.0, 1.0);
}


///////////////////////////////////////////////////////////////
// 屏幕抖动 (Dithering)
//
// 在 8bit 色彩空间中, 细小的颜色渐变会产生肉眼可见的条带。
// 抖动通过在每个像素上添加微小的随机噪声, 打破条带,
// 利用视觉平均产生更平滑的渐变。
//
// 使用 8x8 Bayer 有序抖动矩阵,
// 性能开销远小于 Perlin/Blue noise,
// 且固定模式在相邻帧间稳定, 不会产生闪烁。
///////////////////////////////////////////////////////////////

const DITHER_ELEMENT_RANGE: f32 = 63.0;
const DITHER_LINEAR_COLORSPACE_COLORS: f32 = 256.0;

/// 8x8 Bayer 有序抖动矩阵 [0-63]
/// 来源: https://en.wikipedia.org/wiki/Ordered_dithering
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
 * 根据屏幕坐标计算抖动值 [0, 1]
 *
 * 取像素坐标模 8, 查 8x8 Bayer 矩阵,
 * 确保相邻像素有不同的抖动值,
 * 将量化误差分散到空间邻域。
 *
 * @param gl_FragCoord - 像素屏幕坐标
 */
fn getDitherForPixel(gl_FragCoord: vec2u) -> f32 {
  let pxPos = vec2u(
    gl_FragCoord.x % 8u,
    gl_FragCoord.y % 8u
  );
  let idx = pxPos.y * 8u + pxPos.x;
  // Naga 编译限制: 不能非常量索引 'array<u32, 64>'
  // 已在 'nagaFixes.ts' 中通过类型转换解决
  let matValue = DITHER_MATRIX[idx]; // [1-64]
  return f32(matValue) / DITHER_ELEMENT_RANGE;
}

/**
 * 对颜色施加抖动调制
 * 在颜色值上加一个小的随机偏移, 打破条带
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
// ACES 色调映射 (Tonemapping)
//
// 将 HDR 色彩范围压缩到 SDR [0,1] 范围,
// 同时保持亮部细节和色彩饱和度。
//
// ACES (Academy Color Encoding System) 是目前
// 电影行业标准的色调映射曲线,
// 比 Reinhard / Uncharted2 等老算法色彩还原更准确。
//
// 公式: f(x) = (x*(2.51x+0.03)) / (x*(2.43x+0.59)+0.14)
// 参考: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
///////////////////////////////////////////////////////////////

fn doACES_Tonemapping(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return saturate((x*(a*x+b)) / (x*(c*x+d)+e));
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
      colorMgmt: vec4f,  // .x = gamma, .y = exposure, .z = ditherStr
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

/// Binding 1: 上一 Pass 渲染完成的 HDR 颜色纹理
@group(0) @binding(1)
var _textureSrc: texture_2d<f32>;


///////////////////////////////////////////////////////////////
// 顶点着色器: 全屏三角形
///////////////////////////////////////////////////////////////

@vertex
fn main_vs(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
  return getFullscreenTrianglePosition(VertexIndex);
}


///////////////////////////////////////////////////////////////
// Gamma 校正函数
///////////////////////////////////////////////////////////////

/**
 * Gamma 校正
 * 将线性空间颜色转换到 sRGB 显示空间
 * gamma = 2.2 是标准 sRGB 值
 */
fn doGamma (color: vec3f, gammaValue: f32) -> vec3f {
  return pow(color, vec3f(1.0 / gammaValue));
}


///////////////////////////////////////////////////////////////
// 片段着色器 main_fs
//
// 处理管线 (当 shadingMode == 0 时):
//   1. textureLoad 读取 HDR 颜色
//   2. 抖动 (dither) - 减少条带
//   3. 曝光 (exposure) - 调节亮度
//   4. ACES 色调映射 - HDR → SDR
//   5. Gamma 校正 - 线性 → sRGB
//
// 调试模式: 非 0 模式直接输出原始颜色
//   (用于可视化剔除/法线等调试信息)
///////////////////////////////////////////////////////////////

@fragment
fn main_fs(
  @builtin(position) positionPxF32: vec4<f32>
) -> @location(0) vec4<f32> {
  let fragPositionPx = vec2u(positionPxF32.xy);
  // 从 HDR 纹理中读取颜色 (前一个 Pass 的渲染结果)
  var color = textureLoad(_textureSrc, fragPositionPx, 0).rgb;

  // 只有 shadingMode == 0 时做完整的 Post-processing
  // 其他模式 (调试着色) 直接输出
  let shadingMode = getShadingMode(_uniforms.flags);
  if (shadingMode == 0u) {
    // colorMgmt: .x = gamma, .y = exposure, .z = ditherStr
    let gamma = _uniforms.colorMgmt.x;
    let exposure = _uniforms.colorMgmt.y;
    let ditherStr = _uniforms.colorMgmt.z;

    // 1. 抖动: 在量化前加噪声, 减少条带
    color = ditherColor(fragPositionPx, color, ditherStr);
    // 2. 曝光调整
    color = color * exposure;
    // 3. ACES 色调映射: HDR → SDR [0,1]
    color = saturate(doACES_Tonemapping(color));
    // 4. Gamma 校正: 线性空间 → sRGB
    color = doGamma(color, gamma);
  }

  return vec4(color.xyz, 1.0);
}
