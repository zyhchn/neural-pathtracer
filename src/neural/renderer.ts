// 神经路径追踪渲染器（原生 WebGPU）
//
// 设计要点：
// 1. 相机与交互控制复用 three.js（PerspectiveCamera + OrbitControls），
//    只把它们当参数来源，不参与渲染。
// 2. 渲染本身完全使用原生 WebGPU：一个 compute shader 做路径追踪 / MLP 推理，
//    一个 render shader 把累积结果做色调映射后画到画布上。
// 3. MLP 前向直接在 WGSL 里循环计算，避免 CPU 逐像素推理导致页面卡顿。
//
// 三种模式：
//   compare     —— 左半屏神经结果，右半屏传统结果
//   neural      —— 只画神经结果（主光线 + NEE 直接光 + MLP 间接光）
//   traditional —— 只画传统结果（主光线 + NEE 直接光 + 多 bounce 间接光）

import { MathUtils, PerspectiveCamera, Vector3 } from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { buildRoomSceneData, type RoomSceneData } from "../pathrace/scene-room"
import type { MlpModel } from "./mlp"

/** 三种渲染模式 */
export type NeuralRenderMode = "compare" | "neural" | "traditional"

/** 曝光系数：线性 HDR 值先乘它再进入色调映射 */
const EXPOSURE = 0.2
/** 渲染分辨率缩放：为了性能，按容器尺寸的一半出图 */
const RENDER_SCALE = 0.5
/** MLP 输入维度：位置(3) + 法线(3) + 反照率(3) */
const INPUT_DIM = 9
/** MLP 隐藏层宽度 */
const HIDDEN_DIM = 64
/** MLP 输出维度：RGB 间接光 */
const OUTPUT_DIM = 3
/** 传统模式的最大反弹次数，与训练真值 teacher 保持一致 */
const BOUNCES = 4
/** compute shader 的 workgroup 边长（8x8=64 线程/组） */
const WORKGROUP_SIZE = 8

// ---- MLP 权重在 GPU buffer 中的打包布局 ----
// 所有张量平铺进同一个 f32 数组，用偏移量定位各段。
/** 输入标准化均值段起点 */
const X_MEAN_OFFSET = 0
/** 输入标准化标准差段起点 */
const X_STD_OFFSET = X_MEAN_OFFSET + INPUT_DIM
/** 输出反标准化均值段起点 */
const Y_MEAN_OFFSET = X_STD_OFFSET + INPUT_DIM
/** 输出反标准化标准差段起点 */
const Y_STD_OFFSET = Y_MEAN_OFFSET + OUTPUT_DIM
/** 第 0 层权重段起点 */
const W0_OFFSET = Y_STD_OFFSET + OUTPUT_DIM
/** 第 0 层偏置段起点 */
const B0_OFFSET = W0_OFFSET + INPUT_DIM * HIDDEN_DIM
/** 第 1 层权重段起点 */
const W1_OFFSET = B0_OFFSET + HIDDEN_DIM
/** 第 1 层偏置段起点 */
const B1_OFFSET = W1_OFFSET + HIDDEN_DIM * HIDDEN_DIM
/** 第 2 层权重段起点 */
const W2_OFFSET = B1_OFFSET + HIDDEN_DIM
/** 第 2 层偏置段起点 */
const B2_OFFSET = W2_OFFSET + HIDDEN_DIM * HIDDEN_DIM
/** 第 3 层（输出层）权重段起点 */
const W3_OFFSET = B2_OFFSET + HIDDEN_DIM
/** 第 3 层（输出层）偏置段起点 */
const B3_OFFSET = W3_OFFSET + HIDDEN_DIM * OUTPUT_DIM
/** 打包后整个 MLP buffer 的 f32 元素个数 */
const MLP_BUFFER_SIZE = B3_OFFSET + OUTPUT_DIM

/** 相机参数快照：从 three.js 相机解算出来，再写进 GPU uniform */
type CameraState = {
  /** 相机位置（光线起点） */
  origin: Vector3
  /** 相机朝向（前方向单位向量） */
  forward: Vector3
  /** 相机右方向单位向量 */
  right: Vector3
  /** 相机上方向单位向量 */
  up: Vector3
  /** 竖直半张角的正切值 tan(fov/2) */
  halfHeight: number
  /** 面板宽高比 */
  aspect: number
  /** 面板像素宽 */
  width: number
  /** 面板像素高 */
  height: number
}

/** 创建用于观察场景的交互相机，默认站在房间斜上方看向立方体 */
function makeInteractCamera(width: number, height: number) {
  const camera = new PerspectiveCamera(65, width / height, 0.1, 100)
  camera.position.set(2.4, 2.2, 2.6)
  camera.lookAt(0, 0.9, 0)
  return camera
}

/**
 * 从 three.js 相机矩阵中提取渲染所需参数。
 * 相机矩阵的列向量依次是 right / up / back，取反 back 即得到 forward。
 */
function readCameraState(
  camera: PerspectiveCamera,
  width: number,
  height: number,
): CameraState {
  camera.updateMatrixWorld()

  const e = camera.matrixWorld.elements
  const right = new Vector3(e[0], e[1], e[2])
  const up = new Vector3(e[4], e[5], e[6])
  const back = new Vector3(e[8], e[9], e[10])

  return {
    origin: new Vector3().setFromMatrixPosition(camera.matrixWorld),
    forward: back.negate(),
    right,
    up,
    halfHeight: Math.tan(MathUtils.degToRad(camera.fov) / 2),
    aspect: camera.aspect,
    width,
    height,
  }
}

/** 把渲染模式映射成 shader 里的整数枚举：0=compare, 1=neural, 2=traditional */
function modeToValue(mode: NeuralRenderMode) {
  if (mode === "neural") return 1
  if (mode === "traditional") return 2
  return 0
}

/** 校验模型结构，当前 shader 只支持 9→64→64→64→3 这一种 MLP */
function validateModel(model: MlpModel) {
  const ok =
    model.inDim === INPUT_DIM &&
    model.outDim === OUTPUT_DIM &&
    model.layers.length === 4 &&
    model.layers[0].inDim === INPUT_DIM &&
    model.layers[0].outDim === HIDDEN_DIM &&
    model.layers[1].inDim === HIDDEN_DIM &&
    model.layers[1].outDim === HIDDEN_DIM &&
    model.layers[2].inDim === HIDDEN_DIM &&
    model.layers[2].outDim === HIDDEN_DIM &&
    model.layers[3].inDim === HIDDEN_DIM &&
    model.layers[3].outDim === OUTPUT_DIM

  if (!ok) {
    throw new Error("当前 GPU 版神经渲染器只支持 9→64→64→64→3 的 MLP 结构")
  }
}

/**
 * 创建一个 GPU buffer 并可选用初始数据填充。
 * size 会被向上对齐到 4 字节，满足 WGSL 的对齐要求。
 */
function createGpuBuffer(
  device: GPUDevice,
  usage: GPUBufferUsageFlags,
  dataOrSize: Float32Array | Uint32Array | number,
) {
  const size = typeof dataOrSize === "number" ? dataOrSize : dataOrSize.byteLength
  const buffer = device.createBuffer({
    size: Math.max(4, Math.ceil(size / 4) * 4),
    usage,
  })
  if (typeof dataOrSize !== "number") {
    device.queue.writeBuffer(buffer, 0, dataOrSize)
  }
  return buffer
}

/**
 * 把房间矩形打成 GPU 数据。
 * 每个矩形占 20 个 f32（5 个 vec4）：
 *   [corner.xyz, 0] [edgeU.xyz, 0] [edgeV.xyz, 0] [normal.xyz, 0] [materialId, rectId, 0, 0]
 */
function buildRectBufferData(roomSceneData: RoomSceneData) {
  const rectCount = roomSceneData.rects.length
  const rectStride = 20
  const rectArray = new Float32Array(rectCount * rectStride)

  for (let i = 0; i < rectCount; i++) {
    const rect = roomSceneData.rects[i]
    const materialId = roomSceneData.materialIds[i]
    const rectOffset = i * rectStride
    rectArray.set(rect.corner.toArray(), rectOffset)
    rectArray.set(rect.edgeU.toArray(), rectOffset + 4)
    rectArray.set(rect.edgeV.toArray(), rectOffset + 8)
    rectArray.set(rect.normal.toArray(), rectOffset + 12)
    rectArray.set([materialId, rect.id], rectOffset + 16)
  }

  return rectArray
}

/**
 * 把材质打成 GPU 数据。
 * 每个材质占 8 个 f32（2 个 vec4）：[albedo.xyz, 0] [emission.xyz, 0]
 */
function buildMaterialBufferData(roomSceneData: RoomSceneData) {
  const materialCount = roomSceneData.materials.length
  const materialArray = new Float32Array(materialCount * 8)

  for (let i = 0; i < materialCount; i++) {
    const material = roomSceneData.materials[i]
    const materialOffset = i * 8
    materialArray.set(material.albedo.toArray(), materialOffset)
    materialArray.set(material.emission.toArray(), materialOffset + 4)
  }

  return materialArray
}

/**
 * 把 MLP 的归一化参数与各层权重按固定布局平铺进一个 f32 数组。
 * 布局偏移由上面的 *_OFFSET 常量定义，shader 端按相同偏移读取。
 */
function buildMlpBufferData(model: MlpModel) {
  const packed = new Float32Array(MLP_BUFFER_SIZE)
  packed.set(model.xMean, X_MEAN_OFFSET)
  packed.set(model.xStd, X_STD_OFFSET)
  packed.set(model.yMean, Y_MEAN_OFFSET)
  packed.set(model.yStd, Y_STD_OFFSET)
  packed.set(model.layers[0].w, W0_OFFSET)
  packed.set(model.layers[0].b, B0_OFFSET)
  packed.set(model.layers[1].w, W1_OFFSET)
  packed.set(model.layers[1].b, B1_OFFSET)
  packed.set(model.layers[2].w, W2_OFFSET)
  packed.set(model.layers[2].b, B2_OFFSET)
  packed.set(model.layers[3].w, W3_OFFSET)
  packed.set(model.layers[3].b, B3_OFFSET)
  return packed
}

/**
 * compute shader：逐像素做路径追踪 / MLP 推理，并把结果累加进 accum buffer。
 *
 * 绑定：
 *   binding 0  rects     —— 场景矩形
 *   binding 1  materials —— 材质（反照率 / 自发光）
 *   binding 2  mlp       —— MLP 归一化参数与权重
 *   binding 3  accum     —— 累积结果（可读写）
 *   binding 4  cameraState —— 相机参数
 *   binding 5  settings  —— 运行参数（尺寸 / 帧号 / 模式等）
 */
function createComputeShaderCode() {
  return /* wgsl */ `
struct CameraState {
  origin: vec4f,
  forward: vec4f,
  right: vec4f,
  upHalfHeight: vec4f,
  aspect: vec4f,
}

// 运行参数。u32 共 8 个，正好 32 字节，满足 uniform 对齐。
struct Settings {
  rectCount: u32,
  lightId: u32,
  width: u32,
  height: u32,
  frameIndex: u32,
  mode: u32,
  _pad0: u32,
  _pad1: u32,
}

// 求交结果：是否命中、命中矩形下标、交点距离
struct Hit {
  found: bool,
  rectIndex: u32,
  t: f32,
}

@group(0) @binding(0) var<storage, read> rects: array<vec4f>;
@group(0) @binding(1) var<storage, read> materials: array<vec4f>;
@group(0) @binding(2) var<storage, read> mlp: array<f32>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(4) var<uniform> cameraState: CameraState;
@group(0) @binding(5) var<uniform> settings: Settings;

// 整数哈希，用来把种子扩散成随机数
fn hashU32(x: u32) -> u32 {
  var v = x * 747796405u + 2891336453u;
  v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (v >> 22u) ^ v;
}

// 按种子和维度取一个 [0,1) 浮点随机数
fn randomFloat(seed: u32, dim: u32) -> f32 {
  let value = hashU32(seed + dim * 1013904223u);
  return f32(value) * (1.0 / 4294967295.0);
}

// 由像素坐标与帧号生成该像素这一帧的种子
fn makePixelSeed(px: u32, py: u32, frameIndex: u32) -> u32 {
  return px * 1973u + py * 9277u + frameIndex * 26699u;
}

// SiLU 激活：x * sigmoid(x)
fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

// 余弦加权半球采样：围绕法线取一个方向，返回世界空间单位向量
fn sampleCosineWorld(normal: vec3f, seed: u32) -> vec3f {
  let u1 = randomFloat(seed, 5u);
  let u2 = randomFloat(seed, 6u);
  let r = sqrt(u1);
  let phi = u2 * 6.283185307179586;
  let localDir = vec3f(
    r * cos(phi),
    r * sin(phi),
    sqrt(max(1.0 - u1, 0.0)),
  );

  // 选一个与法线不平行的辅助轴来构造切线空间
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
  let tangent = normalize(cross(up, normal));
  let bitangent = cross(normal, tangent);
  return normalize(tangent * localDir.x + bitangent * localDir.y + normal * localDir.z);
}

// 由像素坐标生成主光线方向（带像素内抖动，用于抗锯齿式累加）
fn generatePrimaryRayDir(px: u32, py: u32, width: u32, height: u32, seed: u32) -> vec3f {
  let u = (f32(px) + randomFloat(seed, 0u)) / f32(width);
  let v = (f32(py) + randomFloat(seed, 1u)) / f32(height);
  let sx = (u * 2.0 - 1.0) * cameraState.aspect.x * cameraState.upHalfHeight.w;
  let sy = (1.0 - v * 2.0) * cameraState.upHalfHeight.w;
  return normalize(cameraState.forward.xyz + cameraState.right.xyz * sx + cameraState.upHalfHeight.xyz * sy);
}

// ---- 矩形字段读取辅助函数：每个矩形占 5 个 vec4 ----
fn rectCorner(rectIndex: u32) -> vec3f {
  return rects[rectIndex * 5u + 0u].xyz;
}
fn rectEdgeU(rectIndex: u32) -> vec3f {
  return rects[rectIndex * 5u + 1u].xyz;
}
fn rectEdgeV(rectIndex: u32) -> vec3f {
  return rects[rectIndex * 5u + 2u].xyz;
}
fn rectNormal(rectIndex: u32) -> vec3f {
  return rects[rectIndex * 5u + 3u].xyz;
}
fn rectMaterialId(rectIndex: u32) -> u32 {
  return u32(rects[rectIndex * 5u + 4u].x);
}
fn rectId(rectIndex: u32) -> u32 {
  return u32(rects[rectIndex * 5u + 4u].y);
}

// ---- 材质字段读取辅助函数：每个材质占 2 个 vec4 ----
fn materialAlbedo(materialId: u32) -> vec3f {
  return materials[materialId * 2u + 0u].xyz;
}
fn materialEmission(materialId: u32) -> vec3f {
  return materials[materialId * 2u + 1u].xyz;
}

// 射线与全部矩形求交，返回最近命中。与 CPU teacher 的参数保护保持一致。
fn intersectScene(rayOrigin: vec3f, rayDir: vec3f) -> Hit {
  var bestT = 1e20;
  var hit = Hit(false, 0u, 0.0);

  for (var rectIndex = 0u; rectIndex < settings.rectCount; rectIndex = rectIndex + 1u) {
    let corner = rectCorner(rectIndex);
    let edgeU = rectEdgeU(rectIndex);
    let edgeV = rectEdgeV(rectIndex);
    let normal = rectNormal(rectIndex);

    // 分母过小说明射线几乎平行于平面，跳过
    let denom = dot(normal, rayDir);
    if (abs(denom) <= 1e-6) {
      continue;
    }

    // t <= 1e-4 用来防自交
    let t = dot(corner - rayOrigin, normal) / denom;
    if (t <= 1e-4 || t >= bestT) {
      continue;
    }

    // 判断交点是否落在矩形参数范围内
    let p = rayOrigin + rayDir * t;
    let rel = p - corner;
    let u = dot(rel, edgeU) / dot(edgeU, edgeU);
    let v = dot(rel, edgeV) / dot(edgeV, edgeV);

    if (u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0) {
      bestT = t;
      hit = Hit(true, rectIndex, t);
    }
  }

  return hit;
}

// 单样本 NEE：在面光源上均匀取点做阴影射线，返回该点的直接光估计
fn sampleDirectLight(point: vec3f, normal: vec3f, albedo: vec3f, seed: u32) -> vec3f {
  let lightRectIndex = settings.lightId;
  let lightCorner = rectCorner(lightRectIndex);
  let lightEdgeU = rectEdgeU(lightRectIndex);
  let lightEdgeV = rectEdgeV(lightRectIndex);
  let lightNormal = rectNormal(lightRectIndex);
  let lightEmission = materialEmission(rectMaterialId(lightRectIndex));

  let lightU = randomFloat(seed, 2u);
  let lightV = randomFloat(seed, 3u);
  let lightPoint = lightCorner + lightEdgeU * lightU + lightEdgeV * lightV;

  let toLight = lightPoint - point;
  let distanceSq = dot(toLight, toLight);
  let wi = normalize(toLight);
  let cp = max(dot(normal, wi), 0.0);
  let cl = max(dot(lightNormal, -wi), 0.0);

  // 任一侧背对光源则没有贡献
  if (cp <= 0.0 || cl <= 0.0) {
    return vec3f(0.0);
  }

  // 阴影射线：顶点必须能直接看到光源
  let shadowOrigin = point + normal * 1e-4;
  let shadowHit = intersectScene(shadowOrigin, wi);
  if (!shadowHit.found || shadowHit.rectIndex != lightRectIndex) {
    return vec3f(0.0);
  }

  // 面积采样 pdf 的倒数 = 光源面积；再乘 BRDF(1/π)
  let lightArea = length(lightEdgeU) * length(lightEdgeV);
  let pdfArea = 1.0 / lightArea;
  let scale = cp * cl / (distanceSq * pdfArea);
  return (albedo / 3.141592653589793) * lightEmission * scale;
}

// Russian Roulette 的存活概率：由路径吞吐量决定，并夹在 [0.2, 0.95]
fn computeContinueProb(beta: vec3f) -> f32 {
  return clamp(max(max(beta.x, beta.y), beta.z), 0.2, 0.95);
}

/**
 * MLP 前向：输入 (位置, 法线, 反照率)，输出预测的间接光 RGB。
 * 结构为 9 → 64 → 64 → 64 → 3，隐藏层用 SiLU，
 * 输出先反标准化再 expm1 还原到线性空间，最后 clamp 到非负。
 */
fn evaluateIndirect(point: vec3f, normal: vec3f, albedo: vec3f) -> vec3f {
  // 输入标准化
  var input: array<f32, ${INPUT_DIM}>;
  input[0] = (point.x - mlp[${X_MEAN_OFFSET}u]) / mlp[${X_STD_OFFSET}u];
  input[1] = (point.y - mlp[${X_MEAN_OFFSET + 1}u]) / mlp[${X_STD_OFFSET + 1}u];
  input[2] = (point.z - mlp[${X_MEAN_OFFSET + 2}u]) / mlp[${X_STD_OFFSET + 2}u];
  input[3] = (normal.x - mlp[${X_MEAN_OFFSET + 3}u]) / mlp[${X_STD_OFFSET + 3}u];
  input[4] = (normal.y - mlp[${X_MEAN_OFFSET + 4}u]) / mlp[${X_STD_OFFSET + 4}u];
  input[5] = (normal.z - mlp[${X_MEAN_OFFSET + 5}u]) / mlp[${X_STD_OFFSET + 5}u];
  input[6] = (albedo.x - mlp[${X_MEAN_OFFSET + 6}u]) / mlp[${X_STD_OFFSET + 6}u];
  input[7] = (albedo.y - mlp[${X_MEAN_OFFSET + 7}u]) / mlp[${X_STD_OFFSET + 7}u];
  input[8] = (albedo.z - mlp[${X_MEAN_OFFSET + 8}u]) / mlp[${X_STD_OFFSET + 8}u];

  // 隐藏层 0：9 → 64
  var hidden0: array<f32, ${HIDDEN_DIM}>;
  for (var o = 0u; o < ${HIDDEN_DIM}u; o = o + 1u) {
    var sum = mlp[${B0_OFFSET}u + o];
    let row = ${W0_OFFSET}u + o * ${INPUT_DIM}u;
    for (var i = 0u; i < ${INPUT_DIM}u; i = i + 1u) {
      sum = sum + mlp[row + i] * input[i];
    }
    hidden0[o] = silu(sum);
  }

  // 隐藏层 1：64 → 64
  var hidden1: array<f32, ${HIDDEN_DIM}>;
  for (var o = 0u; o < ${HIDDEN_DIM}u; o = o + 1u) {
    var sum = mlp[${B1_OFFSET}u + o];
    let row = ${W1_OFFSET}u + o * ${HIDDEN_DIM}u;
    for (var i = 0u; i < ${HIDDEN_DIM}u; i = i + 1u) {
      sum = sum + mlp[row + i] * hidden0[i];
    }
    hidden1[o] = silu(sum);
  }

  // 隐藏层 2：64 → 64
  var hidden2: array<f32, ${HIDDEN_DIM}>;
  for (var o = 0u; o < ${HIDDEN_DIM}u; o = o + 1u) {
    var sum = mlp[${B2_OFFSET}u + o];
    let row = ${W2_OFFSET}u + o * ${HIDDEN_DIM}u;
    for (var i = 0u; i < ${HIDDEN_DIM}u; i = i + 1u) {
      sum = sum + mlp[row + i] * hidden1[i];
    }
    hidden2[o] = silu(sum);
  }

  // 输出层：64 → 3，反标准化 + expm1 还原
  var output = vec3f(0.0);
  for (var o = 0u; o < ${OUTPUT_DIM}u; o = o + 1u) {
    var sum = mlp[${B3_OFFSET}u + o];
    let row = ${W3_OFFSET}u + o * ${HIDDEN_DIM}u;
    for (var i = 0u; i < ${HIDDEN_DIM}u; i = i + 1u) {
      sum = sum + mlp[row + i] * hidden2[i];
    }
    let value = max(exp(sum * mlp[${Y_STD_OFFSET}u + o] + mlp[${Y_MEAN_OFFSET}u + o]) - 1.0, 0.0);
    output[o] = value;
  }

  return output;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x;
  let py = gid.y;
  // 越界线程直接返回
  if (px >= settings.width || py >= settings.height) {
    return;
  }

  // compare 模式横向分两栏：左栏神经，右栏传统
  let compareMode = settings.mode == 0u;
  let traditionalOnly = settings.mode == 2u;
  let panelWidth = select(settings.width, settings.width / 2u, compareMode);
  let isRightPanel = compareMode && px >= panelWidth;
  let localPx = select(px, px - panelWidth, isRightPanel);
  let useTraditional = traditionalOnly || isRightPanel;
  let pixelIndex = py * settings.width + px;
  let seed = makePixelSeed(px, py, settings.frameIndex);
  var rayOrigin = cameraState.origin.xyz;
  var rayDir = generatePrimaryRayDir(localPx, py, panelWidth, settings.height, seed);
  var radiance = vec3f(0.0);

  if (useTraditional) {
    // 传统路径追踪：多个 bounce，每站做 NEE + 余弦采样 + Russian Roulette
    var beta = vec3f(1.0);
    for (var bounce = 0u; bounce < ${BOUNCES}u; bounce = bounce + 1u) {
      let bounceSeed = seed + bounce * 1597334677u;
      let hit = intersectScene(rayOrigin, rayDir);
      if (!hit.found) {
        break;
      }

      let hitPoint = rayOrigin + rayDir * hit.t;
      let normal = rectNormal(hit.rectIndex);
      let materialId = rectMaterialId(hit.rectIndex);
      let objectId = rectId(hit.rectIndex);
      let albedo = materialAlbedo(materialId);
      let emission = materialEmission(materialId);

      // 命中光源：只有主光线直接看到光源时才计入自发光
      if (objectId == settings.lightId) {
        if (bounce == 0u && dot(normal, -rayDir) > 0.0) {
          radiance = radiance + emission;
        }
        break;
      }

      let direct = sampleDirectLight(hitPoint, normal, albedo, bounceSeed);
      radiance = radiance + beta * direct;

      let wi = sampleCosineWorld(normal, bounceSeed);
      beta = beta * albedo;

      // 从第 2 个 bounce 起做俄罗斯轮盘赌
      if (bounce >= 2u) {
        let q = computeContinueProb(beta);
        if (randomFloat(bounceSeed, 4u) > q) {
          break;
        }
        beta = beta / q;
      }

      rayOrigin = hitPoint + normal * 1e-4;
      rayDir = wi;
    }
  } else {
    // 神经路径追踪：只跟一条主光线，命中后 = NEE 直接光 + MLP 预测的间接光
    let hit = intersectScene(rayOrigin, rayDir);
    if (hit.found) {
      let hitPoint = rayOrigin + rayDir * hit.t;
      let normal = rectNormal(hit.rectIndex);
      let materialId = rectMaterialId(hit.rectIndex);
      let objectId = rectId(hit.rectIndex);
      let albedo = materialAlbedo(materialId);
      let emission = materialEmission(materialId);

      if (objectId == settings.lightId) {
        if (dot(normal, -rayDir) > 0.0) {
          radiance = radiance + emission;
        }
      } else {
        let direct = sampleDirectLight(hitPoint, normal, albedo, seed);
        let indirect = evaluateIndirect(hitPoint, normal, albedo);
        radiance = direct + indirect;
      }
    }
  }

  // 累加：frameIndex 为 0 表示刚开始新一轮累积，此时直接覆盖旧值
  let prevSum = accum[pixelIndex].xyz;
  let baseSum = select(prevSum, vec3f(0.0), settings.frameIndex == 0u);
  accum[pixelIndex] = vec4f(baseSum + radiance, 1.0);
}

`
}

/**
 * render shader：全屏三角形，把 accum buffer 的值做平均、色调映射后输出。
 * accum 在 compute 里是可写 storage，这里必须声明为只读 storage，
 * 否则与 render pipeline 的 bind group layout 校验冲突。
 */
function createRenderShaderCode() {
  return /* wgsl */ `
// 与 compute 保持完全一致的布局
struct Settings {
  rectCount: u32,
  lightId: u32,
  width: u32,
  height: u32,
  frameIndex: u32,
  mode: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> accum: array<vec4f>;
@group(0) @binding(1) var<uniform> settings: Settings;

struct VertexOutput {
  @builtin(position) position: vec4f,
}

// 覆盖整个屏幕的大三角形（比两个三角形的四边形更省事）
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  return output;
}

@fragment
fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let x = min(u32(position.x), settings.width - 1u);
  let y = min(u32(position.y), settings.height - 1u);
  let pixelIndex = y * settings.width + x;
  // 除以已累积帧数得到均值
  let sampleCount = f32(settings.frameIndex + 1u);
  let mean = accum[pixelIndex].xyz / sampleCount;
  // 曝光 -> Reinhard 色调映射 -> sRGB 编码
  let exposed = mean * ${EXPOSURE};
  let mapped = exposed / (exposed + vec3f(1.0));
  let srgb = pow(max(mapped, vec3f(0.0)), vec3f(1.0 / 2.2));
  return vec4f(srgb, 1.0);
}
`
}

/**
 * 神经路径追踪渲染器。
 * 构造用静态工厂 create()，因为 WebGPU 设备初始化是异步的。
 */
export class NeuralPathTracer {
  /** 状态回调：页面用它显示「累积样本数」等提示 */
  onStatus: ((text: string) => void) | null = null

  /** 承载画布的容器元素 */
  private container: HTMLElement
  /** 交互相机（只提供参数，不参与渲染） */
  private camera: PerspectiveCamera
  /** 轨道控制器，绑定在画布上 */
  private controls: OrbitControls
  /** 渲染用的画布 */
  private canvas: HTMLCanvasElement
  /** 房间场景数据（矩形 + 材质） */
  private roomSceneData: RoomSceneData
  /** 当前相机参数快照 */
  private cameraState!: CameraState
  /** 监听容器尺寸变化 */
  private resizeObserver: ResizeObserver | null = null

  /** 当前渲染模式 */
  private mode: NeuralRenderMode = "compare"
  /** 整张画布宽（像素） */
  private renderWidth = 1
  /** 整张画布高（像素） */
  private renderHeight = 1
  /** 单个面板宽（compare 模式下为画布一半） */
  private panelWidth = 1
  /** 单个面板高 */
  private panelHeight = 1
  /** 已累积帧数，也是累积权重 */
  private frameIndex = 0
  /** requestAnimationFrame 句柄 */
  private rafId = 0

  // ---- WebGPU 资源 ----
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  private computePipeline!: GPUComputePipeline
  private renderPipeline!: GPURenderPipeline
  private computeBindGroupLayout: any = null
  private renderBindGroupLayout: any = null
  private computeBindGroup!: GPUBindGroup
  private renderBindGroup!: GPUBindGroup

  private rectBuffer!: GPUBuffer
  private materialBuffer!: GPUBuffer
  private mlpBuffer!: GPUBuffer
  private accumBuffer!: GPUBuffer
  private cameraBuffer!: GPUBuffer
  private settingsBuffer!: GPUBuffer

  /** 异步创建并初始化渲染器 */
  static async create(container: HTMLElement, model: MlpModel) {
    const tracer = new NeuralPathTracer(container, model)
    await tracer.init(model)
    return tracer
  }

  private constructor(container: HTMLElement, model: MlpModel) {
    validateModel(model)

    this.container = container
    this.roomSceneData = buildRoomSceneData()

    // 创建画布并铺满容器
    this.canvas = document.createElement("canvas")
    this.canvas.style.display = "block"
    this.canvas.style.width = "100%"
    this.canvas.style.height = "100%"
    this.container.appendChild(this.canvas)

    // 相机与控制器
    this.camera = makeInteractCamera(1, 1)
    this.controls = new OrbitControls(this.camera, this.canvas)
    this.controls.target.set(0, 0.9, 0)
    this.controls.update()
    // 相机一动就重新读取参数并重置累积
    this.controls.addEventListener("change", () => {
      this.updateCameraState()
      this.resetAccumulation()
    })
  }

  /** 初始化 WebGPU：拿设备、建 buffer、建 pipeline */
  private async init(model: MlpModel) {
    if (!navigator.gpu) {
      throw new Error("当前浏览器不支持 WebGPU")
    }

    const gpu = navigator.gpu
    const adapter = await gpu.requestAdapter()
    if (!adapter) {
      throw new Error("无法获取 WebGPU 适配器")
    }

    this.device = await adapter.requestDevice()
    this.context = this.canvas.getContext("webgpu") as GPUCanvasContext
    this.presentationFormat = gpu.getPreferredCanvasFormat()

    // 设备丢失时给页面一个提示
    this.device.lost.then((info) => {
      this.onStatus?.(`WebGPU 设备已丢失：${info.message || info.reason}`)
    }).catch(() => {})

    // 场景相关 buffer（静态，只在上传一次）
    this.rectBuffer = createGpuBuffer(
      this.device,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      buildRectBufferData(this.roomSceneData),
    )
    this.materialBuffer = createGpuBuffer(
      this.device,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      buildMaterialBufferData(this.roomSceneData),
    )
    this.mlpBuffer = createGpuBuffer(
      this.device,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      buildMlpBufferData(model),
    )
    // 相机 uniform：5 个 vec4，共 20 个 f32
    this.cameraBuffer = createGpuBuffer(
      this.device,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      20 * 4,
    )
    // 运行参数 uniform：8 个 u32
    this.settingsBuffer = createGpuBuffer(
      this.device,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      8 * 4,
    )

    // compute 与 render 使用两份 shader，避免 accum 的读写类型冲突
    const computeShaderModule = this.device.createShaderModule({ code: createComputeShaderCode() })
    const renderShaderModule = this.device.createShaderModule({ code: createRenderShaderCode() })

    // compute 的 bind group layout：前三个只读 storage，accum 可写，其余 uniform
    this.computeBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    })
    // render 的 bind group layout：accum 只读 + settings uniform
    this.renderBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    })

    this.computePipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.computeBindGroupLayout],
      }),
      compute: {
        module: computeShaderModule,
        entryPoint: "computeMain",
      },
    })
    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.renderBindGroupLayout],
      }),
      vertex: {
        module: renderShaderModule,
        entryPoint: "vertexMain",
      },
      fragment: {
        module: renderShaderModule,
        entryPoint: "fragmentMain",
        targets: [{ format: this.presentationFormat }],
      },
      primitive: {
        topology: "triangle-list",
      },
    })

    // 跟随容器尺寸变化
    this.resizeObserver = new ResizeObserver(() => this.syncSize())
    this.resizeObserver.observe(this.container)
    this.syncSize()
    this.rafId = window.requestAnimationFrame(this.tick)
  }

  /** 切换渲染模式，并重建尺寸相关资源 */
  setMode(mode: NeuralRenderMode) {
    if (mode === this.mode) return
    this.mode = mode
    this.syncSize()
  }

  getMode() {
    return this.mode
  }

  /** 释放所有 GPU 资源 */
  dispose() {
    window.cancelAnimationFrame(this.rafId)
    this.resizeObserver?.disconnect()
    this.controls.dispose()
    this.rectBuffer?.destroy()
    this.materialBuffer?.destroy()
    this.mlpBuffer?.destroy()
    this.accumBuffer?.destroy()
    this.cameraBuffer?.destroy()
    this.settingsBuffer?.destroy()
    this.canvas.remove()
  }

  /** 重新读取相机参数 */
  private updateCameraState() {
    this.cameraState = readCameraState(this.camera, this.panelWidth, this.panelHeight)
    this.camera.aspect = this.panelWidth / this.panelHeight
    this.camera.updateProjectionMatrix()
  }

  /** 把相机参数写进 uniform buffer（5 个 vec4） */
  private writeCameraBuffer() {
    const data = new Float32Array(20)
    data.set([
      this.cameraState.origin.x, this.cameraState.origin.y, this.cameraState.origin.z, 0,
      this.cameraState.forward.x, this.cameraState.forward.y, this.cameraState.forward.z, 0,
      this.cameraState.right.x, this.cameraState.right.y, this.cameraState.right.z, 0,
      this.cameraState.up.x, this.cameraState.up.y, this.cameraState.up.z, this.cameraState.halfHeight,
      this.cameraState.aspect, 0, 0, 0,
    ])
    this.device.queue.writeBuffer(this.cameraBuffer, 0, data)
  }

  /** 把运行参数写进 uniform buffer（8 个 u32） */
  private writeSettingsBuffer() {
    const data = new Uint32Array(8)
    data[0] = this.roomSceneData.rects.length
    data[1] = this.roomSceneData.lightRectId
    data[2] = this.renderWidth
    data[3] = this.renderHeight
    data[4] = this.frameIndex
    data[5] = modeToValue(this.mode)
    this.device.queue.writeBuffer(this.settingsBuffer, 0, data)
  }

  /** 重置累积，重新从第 0 帧开始采样 */
  private resetAccumulation() {
    this.frameIndex = 0
    this.onStatus?.("GPU 累积样本 0")
  }

  /**
   * 同步画布尺寸与布局。
   * compare 模式下画布横向切分为两个面板，面板宽度为画布一半。
   */
  private syncSize() {
    const rect = this.container.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    const width = Math.max(1, Math.round(rect.width * dpr * RENDER_SCALE))
    const height = Math.max(1, Math.round(rect.height * dpr * RENDER_SCALE))
    const columns = this.mode === "compare" ? 2 : 1
    const panelWidth = Math.max(1, Math.floor(width / columns))
    const renderWidth = panelWidth * columns

    // 尺寸没变时只需重置累积
    if (
      renderWidth === this.renderWidth &&
      height === this.renderHeight &&
      panelWidth === this.panelWidth
    ) {
      this.resetAccumulation()
      return
    }

    this.renderWidth = renderWidth
    this.renderHeight = height
    this.panelWidth = panelWidth
    this.panelHeight = height

    this.canvas.width = this.renderWidth
    this.canvas.height = this.renderHeight
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "opaque",
    })

    this.camera.aspect = this.panelWidth / this.panelHeight
    this.camera.updateProjectionMatrix()
    this.cameraState = readCameraState(this.camera, this.panelWidth, this.panelHeight)

    this.rebuildFrameResources()
    this.resetAccumulation()
  }

  /** 尺寸变化后重建依赖画布大小的资源：累积 buffer 与两个 bind group */
  private rebuildFrameResources() {
    // 每个像素一个 vec4，共 width*height*4 个 f32
    this.accumBuffer?.destroy()
    this.accumBuffer = createGpuBuffer(
      this.device,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      this.renderWidth * this.renderHeight * 4 * 4,
    )

    this.computeBindGroup = this.device.createBindGroup({
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.rectBuffer } },
        { binding: 1, resource: { buffer: this.materialBuffer } },
        { binding: 2, resource: { buffer: this.mlpBuffer } },
        { binding: 3, resource: { buffer: this.accumBuffer } },
        { binding: 4, resource: { buffer: this.cameraBuffer } },
        { binding: 5, resource: { buffer: this.settingsBuffer } },
      ],
    })

    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.accumBuffer } },
        { binding: 1, resource: { buffer: this.settingsBuffer } },
      ],
    })
  }

  /** 每帧：跑一次 compute 采样，再把累积结果渲染到画布 */
  private tick = () => {
    this.controls.update()
    this.cameraState = readCameraState(this.camera, this.panelWidth, this.panelHeight)
    this.writeCameraBuffer()
    this.writeSettingsBuffer()

    const encoder = this.device.createCommandEncoder()

    // compute pass：对每个像素采样一次
    const computePass = encoder.beginComputePass()
    computePass.setPipeline(this.computePipeline)
    computePass.setBindGroup(0, this.computeBindGroup)
    computePass.dispatchWorkgroups(
      Math.ceil(this.renderWidth / WORKGROUP_SIZE),
      Math.ceil(this.renderHeight / WORKGROUP_SIZE),
    )
    computePass.end()

    // render pass：全屏三角形输出
    const view = this.context.getCurrentTexture().createView()
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 1],
        },
      ],
    })
    renderPass.setPipeline(this.renderPipeline)
    renderPass.setBindGroup(0, this.renderBindGroup)
    renderPass.draw(3)
    renderPass.end()

    this.device.queue.submit([encoder.finish()])

    this.frameIndex += 1
    this.onStatus?.(`GPU 累积样本 ${this.frameIndex}`)
    this.rafId = window.requestAnimationFrame(this.tick)
  }
}
