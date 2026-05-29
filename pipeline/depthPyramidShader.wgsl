///////////////////////////////////////////////////////////////
// 深度金字塔 Shader (Depth Pyramid / Hierarchical Z-Buffer)
//
// 用途: 将上一帧的深度纹理降低一个 mip 层级
//
// 深度金字塔在遮挡剔除中的作用:
//   遮挡剔除需要判断"一个包围球是否被已渲染的几何体挡住"。
//   如果在全分辨率深度图上做这个查询，每个包围球需要采样
//   一个像素区域内的多个深度值。对小包围球来说，大部分像素
//   的深度关系不大，采样太多浪费。
//
//   深度金字塔通过逐层降低分辨率，预计算每个区域的最大深度值，
//   让遮挡剔除可以在合适的 mipmap 层级上只采样一个 texel，
//   就能知道"这个包围球投影区域的最浅深度是多少"。
//
// 算法:
//   每个线程负责将 2×2 的 4 个像素合并为 1 个像素。
//   取 4 个深度的最大值 (max)。因为深度比较模式是 'less',
//   越小的值表示越近。取最大值 = 取该区域的最深值 (保守策略)。
//
// 执行方式:
//   上一层 → 本 shader → 下一层
//   dispatch 多次, 每次降低一半分辨率, 直到 1×1。
//   通常 4-6 层 (取决于视口尺寸) 就足够遮挡剔除使用。
//
// 参考: https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-5-using-depth-pyramids-directx-9
///////////////////////////////////////////////////////////////


/// Binding 0: 上一层的深度纹理 (只读)
/// 全分辨率或上一 mip 层级的深度图
/// 每个 texel = 单个 float (r32float)
@group(0) @binding(0)
var _textureSrc: texture_2d<f32>;

/// Binding 1: 这一层的深度纹理 (只写 storage texture)
/// 分辨率 = src 的一半 (向下取整)
/// 类型 r32float, 存储合并后的单一深度值
@group(0) @binding(1)
var _textureDst: texture_storage_2d<r32float, write>;


/**
 * Compute Shader 主入口
 *
 * 线程分配:
 *   @workgroup_size(8, 8, 1) → 每个工作组 64 个线程
 *   global_id.xy = 输出纹理中的像素坐标
 *
 * 每个线程输出 1 个像素 (取 2×2 区域的最大深度)
 */
@compute
@workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  // 当前输出像素坐标
  let index = global_id.xy;

  // 获取上一层纹理尺寸
  let dimSrc = vec2u(textureDimensions(_textureSrc, 0));

  // 边界检查: 确保不超出源纹理范围
  if (index.x >= dimSrc.x || index.y >= dimSrc.y){
    return;
  }

  // 读取 2×2 区域的 4 个深度值, 取最大值
  //
  // 为什么用 max 而不是 min?
  //   WebGPU 默认深度比较为 'less': 近处 = 小值。
  //   遮挡剔除判断: "这个包围球是否完全被更近的物体挡住?"
  //   如果包围球投影区域内的某个像素深度比包围球深度更小 (更近),
  //   则包围球被部分/完全遮挡。
  //
  //   取 max (最深值) 是保守策略:
  //     如果包围球比这个区域的最深值还远 → 肯定被挡住
  //     如果包围球比这个区域的最浅值还近 → 要绘制
  //     如果包围球在 最浅值 < 包围球 < 最深值 → 部分可见, 也可保守剔除
  //
  //   即: 我们假设整个 2×2 区域都被最深值覆盖,
  //       只要包围球比这个最深值还远, 就安全剔除。
  //       这可能会多画一些 (保守), 但不会误剔除。
  //
  // 性能说明: 理论上可以用 textureGather() 一次性读取 4 个值,
  // 但 WGSL 的限制导致这里用 4 次 textureLoad。
  var depth = 0.0;
  let pos = index * 2;

  depth = max(depth, textureLoad(_textureSrc, pos                , 0).x);
  depth = max(depth, textureLoad(_textureSrc, pos + vec2u(0u, 1u), 0).x);
  depth = max(depth, textureLoad(_textureSrc, pos + vec2u(1u, 0u), 0).x);
  depth = max(depth, textureLoad(_textureSrc, pos + vec2u(1u, 1u), 0).x);

  // 写入输出纹理
  textureStore(_textureDst, index, vec4(depth));
}
