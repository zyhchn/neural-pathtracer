// 浏览器侧的 MLP 推理：加载 JSON 权重，做一次前向传播。
//
// 权重来自 training/export-weights.py，与训练时的结构、归一化参数完全一致：
//   x -> 标准化 -> [Linear + SiLU] * N -> Linear -> 反标准化 + expm1 -> clamp(min=0)

export type MlpLayerData = {
  inDim: number
  outDim: number
  /** 行优先展平，索引 = out * inDim + i */
  w: Float32Array
  b: Float32Array
}

export type MlpModel = {
  inDim: number
  outDim: number
  xMean: Float32Array
  xStd: Float32Array
  yMean: Float32Array
  yStd: Float32Array
  layers: MlpLayerData[]
}

type RawLayer = { inDim: number; outDim: number; w: number[]; b: number[] }

type RawModel = {
  inDim: number
  outDim: number
  xMean: number[]
  xStd: number[]
  yMean: number[]
  yStd: number[]
  weights: RawLayer[]
}

export async function loadMlpModel(url: string): Promise<MlpModel> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`加载模型失败：HTTP ${response.status} ${url}`)
  }
  const raw = (await response.json()) as RawModel

  return {
    inDim: raw.inDim,
    outDim: raw.outDim,
    xMean: Float32Array.from(raw.xMean),
    xStd: Float32Array.from(raw.xStd),
    yMean: Float32Array.from(raw.yMean),
    yStd: Float32Array.from(raw.yStd),
    layers: raw.weights.map((layer) => ({
      inDim: layer.inDim,
      outDim: layer.outDim,
      w: Float32Array.from(layer.w),
      b: Float32Array.from(layer.b),
    })),
  }
}

export type MlpEvaluator = (input: ArrayLike<number>, out: Float32Array) => void

/** 创建一个带复用缓冲的前向函数，避免每像素分配内存 */
export function createMlpEvaluator(model: MlpModel): MlpEvaluator {
  let maxDim = Math.max(model.inDim, model.outDim)
  for (const layer of model.layers) {
    maxDim = Math.max(maxDim, layer.outDim)
  }

  const normalized = new Float32Array(maxDim)
  const bufferA = new Float32Array(maxDim)
  const bufferB = new Float32Array(maxDim)

  return function evaluate(input: ArrayLike<number>, out: Float32Array) {
    for (let i = 0; i < model.inDim; i++) {
      normalized[i] = (input[i] - model.xMean[i]) / model.xStd[i]
    }

    let current = normalized
    let target = bufferA
    const lastLayer = model.layers.length - 1

    for (let l = 0; l < model.layers.length; l++) {
      const layer = model.layers[l]
      const isLast = l === lastLayer
      for (let o = 0; o < layer.outDim; o++) {
        let sum = layer.b[o]
        const row = o * layer.inDim
        for (let i = 0; i < layer.inDim; i++) {
          sum += layer.w[row + i] * current[i]
        }
        // SiLU：x * sigmoid(x)
        target[o] = isLast ? sum : sum / (1 + Math.exp(-sum)) * sum
      }
      current = target
      target = target === bufferA ? bufferB : bufferA
    }

    for (let c = 0; c < model.outDim; c++) {
      const value = Math.expm1(current[c] * model.yStd[c] + model.yMean[c])
      out[c] = value > 0 ? value : 0
    }
  }
}
