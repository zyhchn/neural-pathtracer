declare global {
type GPUSize64 = number
type GPUIntegerCoordinate = number
type GPUIndex32 = number
type GPUBufferDynamicOffset = number
type GPUFlagsConstant = number
type GPUColor = [number, number, number, number]

type GPUBufferUsageFlags = GPUFlagsConstant
type GPUTextureUsageFlags = GPUFlagsConstant
type GPUShaderStageFlags = GPUFlagsConstant

type GPUTextureFormat = string
type GPUTextureViewDimension = string
type GPUTextureDimension = '1d' | '2d' | '3d'
type GPUTextureAspect = 'all' | 'stencil-only' | 'depth-only'
type GPUAddressMode = 'clamp-to-edge' | 'repeat' | 'mirror-repeat'
type GPUFilterMode = 'nearest' | 'linear'
type GPUMipmapFilterMode = 'nearest' | 'linear'
type GPUCompareFunction =
  | 'never'
  | 'less'
  | 'equal'
  | 'less-equal'
  | 'greater'
  | 'not-equal'
  | 'greater-equal'
  | 'always'
type GPULoadOp = 'load' | 'clear'
type GPUStoreOp = 'store' | 'discard'

interface GPUObjectBase {
  label?: string
}

interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>
  getPreferredCanvasFormat(): GPUTextureFormat
}

interface GPURequestAdapterOptions {
  powerPreference?: 'low-power' | 'high-performance'
  forceFallbackAdapter?: boolean
}

interface GPUAdapterInfo {
  vendor?: string
  architecture?: string
  device?: string
  description?: string
}

interface GPUAdapter extends GPUObjectBase {
  features: Set<string>
  limits: Record<string, number>
  info?: GPUAdapterInfo
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>
}

interface GPUDeviceDescriptor extends GPUObjectBase {
  requiredFeatures?: string[]
  requiredLimits?: Record<string, number>
}

interface GPUDevice extends GPUObjectBase {
  readonly queue: GPUQueue
  readonly lost: Promise<any>
  destroy(): void
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer
  createTexture(descriptor: GPUTextureDescriptor): GPUTexture
  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder
}

interface GPUQueue {
  submit(commandBuffers: GPUCommandBuffer[]): void
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: GPUSize64,
    data: AllowSharedBufferSource,
    dataOffset?: number,
    size?: number,
  ): void
  writeTexture(
    destination: GPUImageCopyTexture,
    data: AllowSharedBufferSource,
    dataLayout: GPUImageDataLayout,
    size: GPUExtent3D,
  ): void
}

interface GPUBufferDescriptor extends GPUObjectBase {
  size: GPUSize64
  usage: GPUBufferUsageFlags
  mappedAtCreation?: boolean
}

interface GPUBuffer extends GPUObjectBase {
  readonly size: GPUSize64
  readonly usage: GPUBufferUsageFlags
  destroy(): void
  getMappedRange(offset?: GPUSize64, size?: GPUSize64): ArrayBuffer
  unmap(): void
}

interface GPUTextureDescriptor extends GPUObjectBase {
  size: GPUExtent3D
  mipLevelCount?: number
  sampleCount?: number
  dimension?: GPUTextureDimension
  format: GPUTextureFormat
  usage: GPUTextureUsageFlags
}

interface GPUTexture extends GPUObjectBase {
  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView
  destroy(): void
}

interface GPUTextureView extends GPUObjectBase {}

interface GPUTextureViewDescriptor extends GPUObjectBase {
  format?: GPUTextureFormat
  dimension?: GPUTextureViewDimension
  aspect?: GPUTextureAspect
  baseMipLevel?: number
  mipLevelCount?: number
  baseArrayLayer?: number
  arrayLayerCount?: number
}

interface GPUSamplerDescriptor extends GPUObjectBase {
  addressModeU?: GPUAddressMode
  addressModeV?: GPUAddressMode
  addressModeW?: GPUAddressMode
  magFilter?: GPUFilterMode
  minFilter?: GPUFilterMode
  mipmapFilter?: GPUMipmapFilterMode
  compare?: GPUCompareFunction
}

interface GPUSampler extends GPUObjectBase {}

interface GPUShaderModuleDescriptor extends GPUObjectBase {
  code: string
}

interface GPUShaderModule extends GPUObjectBase {}

interface GPUBindGroupLayoutDescriptor extends GPUObjectBase {
  entries: GPUBindGroupLayoutEntry[]
}

interface GPUBindGroupLayoutEntry {
  binding: number
  visibility: GPUShaderStageFlags
  buffer?: GPUBufferBindingLayout
  sampler?: GPUSamplerBindingLayout
  texture?: GPUTextureBindingLayout
  storageTexture?: GPUStorageTextureBindingLayout
}

interface GPUBufferBindingLayout {
  type?: 'uniform' | 'storage' | 'read-only-storage'
  hasDynamicOffset?: boolean
  minBindingSize?: GPUSize64
}

interface GPUSamplerBindingLayout {
  type?: 'filtering' | 'non-filtering' | 'comparison'
}

interface GPUTextureBindingLayout {
  sampleType?: 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint'
  viewDimension?: GPUTextureViewDimension
  multisampled?: boolean
}

interface GPUStorageTextureBindingLayout {
  access?: 'write-only' | 'read-only' | 'read-write'
  format: GPUTextureFormat
  viewDimension?: GPUTextureViewDimension
}

interface GPUBindGroupDescriptor extends GPUObjectBase {
  layout: GPUBindGroupLayout
  entries: GPUBindGroupEntry[]
}

interface GPUBindGroupEntry {
  binding: number
  resource:
    | GPUSampler
    | GPUTextureView
    | GPUBufferBinding
}

interface GPUBufferBinding {
  buffer: GPUBuffer
  offset?: GPUSize64
  size?: GPUSize64
}

interface GPUBindGroup extends GPUObjectBase {}

interface GPUPipelineLayoutDescriptor extends GPUObjectBase {
  bindGroupLayouts: GPUBindGroupLayout[]
}

interface GPUPipelineLayout extends GPUObjectBase {}

interface GPUProgrammableStage {
  module: GPUShaderModule
  entryPoint: string
}

interface GPUComputePipelineDescriptor extends GPUObjectBase {
  layout: GPUPipelineLayout | 'auto'
  compute: GPUProgrammableStage
}

interface GPUComputePipeline extends GPUObjectBase {}

interface GPURenderPipelineDescriptor extends GPUObjectBase {
  layout: GPUPipelineLayout | 'auto'
  vertex: GPUVertexState
  fragment?: GPUFragmentState
  primitive?: GPUPrimitiveState
  depthStencil?: GPUDepthStencilState
  multisample?: GPUMultisampleState
}

interface GPUVertexState extends GPUProgrammableStage {
  buffers?: GPUVertexBufferLayout[]
}

interface GPUVertexBufferLayout {
  arrayStride: GPUSize64
  stepMode?: 'vertex' | 'instance'
  attributes: GPUVertexAttribute[]
}

interface GPUVertexAttribute {
  format: string
  offset: GPUSize64
  shaderLocation: number
}

interface GPUFragmentState extends GPUProgrammableStage {
  targets: Array<GPUColorTargetState | null>
}

interface GPUColorTargetState {
  format: GPUTextureFormat
  blend?: GPUBlendState
  writeMask?: number
}

interface GPUBlendState {
  color: GPUBlendComponent
  alpha: GPUBlendComponent
}

interface GPUBlendComponent {
  operation?: 'add' | 'subtract' | 'reverse-subtract' | 'min' | 'max'
  srcFactor?: string
  dstFactor?: string
}

interface GPUPrimitiveState {
  topology?: 'point-list' | 'line-list' | 'line-strip' | 'triangle-list' | 'triangle-strip'
  stripIndexFormat?: 'uint16' | 'uint32'
  frontFace?: 'ccw' | 'cw'
  cullMode?: 'none' | 'front' | 'back'
}

interface GPUDepthStencilState {
  format: GPUTextureFormat
  depthWriteEnabled?: boolean
  depthCompare?: GPUCompareFunction
}

interface GPUMultisampleState {
  count?: number
  mask?: number
  alphaToCoverageEnabled?: boolean
}

interface GPURenderPipeline extends GPUObjectBase {}

interface GPUCommandEncoderDescriptor extends GPUObjectBase {}

interface GPUCommandEncoder extends GPUObjectBase {
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: GPUSize64,
    destination: GPUBuffer,
    destinationOffset: GPUSize64,
    size: GPUSize64,
  ): void
}

interface GPUComputePassDescriptor extends GPUObjectBase {}

interface GPUComputePassEncoder extends GPUObjectBase {
  setPipeline(pipeline: GPUComputePipeline): void
  setBindGroup(
    index: GPUIndex32,
    bindGroup: GPUBindGroup,
    dynamicOffsets?: Iterable<GPUBufferDynamicOffset>,
  ): void
  dispatchWorkgroups(
    workgroupCountX: GPUIntegerCoordinate,
    workgroupCountY?: GPUIntegerCoordinate,
    workgroupCountZ?: GPUIntegerCoordinate,
  ): void
  end(): void
}

interface GPURenderPassDescriptor extends GPUObjectBase {
  colorAttachments: Array<GPURenderPassColorAttachment | null>
  depthStencilAttachment?: GPURenderPassDepthStencilAttachment
}

interface GPURenderPassColorAttachment {
  view: GPUTextureView
  resolveTarget?: GPUTextureView
  clearValue?: GPUColor
  loadOp: GPULoadOp
  storeOp: GPUStoreOp
}

interface GPURenderPassDepthStencilAttachment {
  view: GPUTextureView
  depthClearValue?: number
  depthLoadOp?: GPULoadOp
  depthStoreOp?: GPUStoreOp
  stencilClearValue?: number
  stencilLoadOp?: GPULoadOp
  stencilStoreOp?: GPUStoreOp
}

interface GPURenderPassEncoder extends GPUObjectBase {
  setPipeline(pipeline: GPURenderPipeline): void
  setBindGroup(
    index: GPUIndex32,
    bindGroup: GPUBindGroup,
    dynamicOffsets?: Iterable<GPUBufferDynamicOffset>,
  ): void
  setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: GPUSize64, size?: GPUSize64): void
  draw(
    vertexCount: number,
    instanceCount?: number,
    firstVertex?: number,
    firstInstance?: number,
  ): void
  end(): void
}

interface GPUCommandBufferDescriptor extends GPUObjectBase {}

interface GPUCommandBuffer extends GPUObjectBase {}

interface GPUCanvasConfiguration {
  device: GPUDevice
  format: GPUTextureFormat
  usage?: GPUTextureUsageFlags
  viewFormats?: GPUTextureFormat[]
  colorSpace?: 'srgb' | 'display-p3'
  alphaMode?: 'opaque' | 'premultiplied'
}

interface GPUCanvasContext {
  configure(configuration: GPUCanvasConfiguration): void
  unconfigure(): void
  getCurrentTexture(): GPUTexture
}

interface GPUImageCopyTexture {
  texture: GPUTexture
  mipLevel?: number
  origin?: GPUOrigin3D
  aspect?: GPUTextureAspect
}

interface GPUImageDataLayout {
  offset?: GPUSize64
  bytesPerRow?: number
  rowsPerImage?: number
}

interface GPUExtent3DDict {
  width: number
  height?: number
  depthOrArrayLayers?: number
}

type GPUExtent3D = [number, number?, number?] | GPUExtent3DDict
type GPUOrigin3D = [number, number?, number?] | { x?: number; y?: number; z?: number }

declare var GPUBufferUsage: {
  readonly MAP_READ: GPUBufferUsageFlags
  readonly MAP_WRITE: GPUBufferUsageFlags
  readonly COPY_SRC: GPUBufferUsageFlags
  readonly COPY_DST: GPUBufferUsageFlags
  readonly INDEX: GPUBufferUsageFlags
  readonly VERTEX: GPUBufferUsageFlags
  readonly UNIFORM: GPUBufferUsageFlags
  readonly STORAGE: GPUBufferUsageFlags
  readonly INDIRECT: GPUBufferUsageFlags
  readonly QUERY_RESOLVE: GPUBufferUsageFlags
}

declare var GPUTextureUsage: {
  readonly COPY_SRC: GPUTextureUsageFlags
  readonly COPY_DST: GPUTextureUsageFlags
  readonly TEXTURE_BINDING: GPUTextureUsageFlags
  readonly STORAGE_BINDING: GPUTextureUsageFlags
  readonly RENDER_ATTACHMENT: GPUTextureUsageFlags
}

declare var GPUShaderStage: {
  readonly VERTEX: GPUShaderStageFlags
  readonly FRAGMENT: GPUShaderStageFlags
  readonly COMPUTE: GPUShaderStageFlags
}

  interface Navigator {
    readonly gpu?: GPU
  }

  interface HTMLCanvasElement {
    getContext(contextId: 'webgpu'): GPUCanvasContext | null
  }
}

export {}
