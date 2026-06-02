import { NaniteObject } from '../../scene/naniteObject.ts';
import { assertIsGPUTextureView } from '../../utils/webgpu.ts';
import {
  BindingsCache,
  assignResourcesToBindings2,
  labelPipeline,
  labelShader,
} from '../_shared.ts';
import { PassCtx } from '../passCtx.ts';
import { SHADER_PARAMS, SHADER_CODE } from './cullInstancesPass.wgsl.ts';

export class CullInstancesPass {
  public static NAME: string = 'CullInstancesPass';

  private readonly pipeline: GPUComputePipeline;
  private readonly bindingsCache = new BindingsCache();

  constructor(device: GPUDevice) {
    const shaderModule = device.createShaderModule({
      label: labelShader(CullInstancesPass),
      code: SHADER_CODE(),
    });

    this.pipeline = device.createComputePipeline({
      label: labelPipeline(CullInstancesPass),
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  onViewportResize = () => {
    this.bindingsCache.clear();
  };

  cmdCullInstances(ctx: PassCtx, naniteObject: NaniteObject) {
    const { cmdBuf, profiler } = ctx;

    // forget draws from previous frame
    // clear bunny-nanite-drawn-instances-ids
    naniteObject.buffers.cmdClearDrawnInstancesDispatchParams(cmdBuf);
    // clear bunny-nanite-billboards
    naniteObject.buffers.cmdClearDrawnImpostorsParams(cmdBuf);

    const computePass = cmdBuf.beginComputePass({
      label: CullInstancesPass.NAME,
      timestampWrites: profiler?.createScopeGpu(CullInstancesPass.NAME),
    });

    const pipeline = this.pipeline;
    // 绑定instance裁剪所需的资源
    const bindings = this.bindingsCache.getBindings(naniteObject.name, () =>
      this.createBindings(ctx, pipeline, naniteObject)
    );

    // 设置管线
    computePass.setPipeline(pipeline);
    // 绑定资源，都绑定group(0)上
    computePass.setBindGroup(0, bindings);

    // dispatch params
    // X: one per instance, but do not overflow limit 65k
    const workgroupsCntX = Math.min(
      naniteObject.instancesCount,
      SHADER_PARAMS.maxWorkgroupsY
    );

    const workgroupsCntY = 1;
    const workgroupsCntZ = 1;

    // dispatch
    // console.log(`${CullInstancesPass.NAME} dispatch(${workgroupsCntX}, ${workgroupsCntY}, ${workgroupsCntZ})`); // prettier-ignore
    // 启动线程进行计算
    computePass.dispatchWorkgroups(
      workgroupsCntX,
      workgroupsCntY,
      workgroupsCntZ
    );

    // 管线结束
    computePass.end();
  }

  private createBindings = (
    {
      device,
      globalUniforms,
      prevFrameDepthPyramidTexture,
      depthPyramidSampler,
    }: PassCtx,
    pipeline: GPUComputePipeline,
    naniteObject: NaniteObject
  ): GPUBindGroup => {
    const b = SHADER_PARAMS.bindings;
    assertIsGPUTextureView(prevFrameDepthPyramidTexture);

    const buffers = naniteObject.buffers;

    return assignResourcesToBindings2(
      CullInstancesPass,
      naniteObject.name,
      device,
      pipeline,
      [
        globalUniforms.createBindingDesc(b.renderUniforms),
        naniteObject.bindInstanceTransforms(b.instancesTransforms),
        // 绑定_drawnInstancesParams
        buffers.bindDrawnInstancesParams(b.dispatchIndirectParams), 
        // 绑定_drawnInstancesList
        buffers.bindDrawnInstancesList(b.drawnInstanceIdsResult),
        // 绑定_drawnImpostorsParams
        buffers.bindDrawnImpostorsParams(b.billboardsParams),
        // 绑定_drawnImpostorsList
        buffers.bindDrawnImpostorsList(b.billboardsIdsResult),
        {
          binding: b.depthPyramidTexture,
          resource: prevFrameDepthPyramidTexture,
        },
        { 
          binding: b.depthSampler, 
          resource: depthPyramidSampler 
        },
      ]
    );
  };
}
