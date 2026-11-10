// 训练数据生成脚本
//
// 作用：在固定场景上采样训练点，用 CPU teacher 算出「间接光」真值，写出数据集；
// 可选地额外生成一份像素探针数据，供 Python 侧画 A/B 对照图。
//
// 用法（在项目根目录执行）：
//   node training/generate-dataset.ts --count 8000 --spp 512 --sample-mode balanced --with-pixels
//
// 产物：
//   training/data/indirect_dataset.jsonl
//   training/data/pixel_probe.json（开启 --with-pixels 时）
//
// 该脚本只被 Node 直接运行，不参与前端构建。

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { PerspectiveCamera } from "three"
import { buildRoomSceneData } from "../src/pathrace/scene-room.ts"
import {
  blackenNonEmissiveAlbedo,
  buildTeacherScene,
  intersectScene,
  mulberry32,
  sampleDirectLight,
  traceIndirect,
} from "../src/neural/tracer-core.ts"
import type { TeacherScene, Vec3 } from "../src/neural/tracer-core.ts"

const HERE = import.meta.dirname

type Args = Map<string, string | boolean>

function readArgs(argv: string[]): Args {
  const args: Args = new Map()
  for (let i = 0; i < argv.length; i++) {
    const raw = argv[i]
    if (!raw.startsWith("--")) continue
    const key = raw.slice(2)
    const next = argv[i + 1]
    if (next === undefined || next.startsWith("--")) {
      args.set(key, true)
    } else {
      args.set(key, next)
      i++
    }
  }
  return args
}

function numArg(args: Args, key: string, fallback: number) {
  const value = args.get(key)
  if (typeof value !== "string") return fallback
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function intArg(args: Args, key: string, fallback: number) {
  return Math.round(numArg(args, key, fallback))
}

function strArg(args: Args, key: string, fallback: string) {
  const value = args.get(key)
  return typeof value === "string" ? value : fallback
}

function flagArg(args: Args, key: string) {
  return args.get(key) === true
}

function round(value: number) {
  return Number(value.toFixed(6))
}

function roundVec(v: Vec3): [number, number, number] {
  return [round(v.x), round(v.y), round(v.z)]
}

const args = readArgs(process.argv.slice(2))
const count = intArg(args, "count", 8000)
const spp = intArg(args, "spp", 512)
const seed = intArg(args, "seed", 20260101)
const outDir = resolve(strArg(args, "out", resolve(HERE, "data")))
const withPixels = flagArg(args, "with-pixels")
const pixelWidth = intArg(args, "pixel-width", 160)
const pixelHeight = intArg(args, "pixel-height", 120)
const pixelSpp = intArg(args, "pixel-spp", 64)
const blackAlbedo = flagArg(args, "black-albedo")
const sampleMode = strArg(args, "sample-mode", "area")

const scene: TeacherScene = buildTeacherScene(buildRoomSceneData())
if (blackAlbedo) {
  blackenNonEmissiveAlbedo(scene)
  console.log("[自检] 已把所有非发光反照率置零，间接光标签应全部为 0")
}

mkdirSync(outDir, { recursive: true })

/** 按面积加权随机挑一个可采样矩形 */
function pickRect(rand: () => number, totalArea: number) {
  let threshold = rand() * totalArea
  for (let i = 0; i < scene.sampleableIndices.length; i++) {
    threshold -= scene.sampleableAreas[i]
    if (threshold <= 0) return scene.sampleableIndices[i]
  }
  return scene.sampleableIndices[scene.sampleableIndices.length - 1]
}

/** 按面均衡挑一个可采样矩形，让小物体也拿到足够监督 */
function pickRectBalanced(sampleIndex: number) {
  return scene.sampleableIndices[sampleIndex % scene.sampleableIndices.length]
}

function generateDataset() {
  const totalArea = scene.sampleableAreas.reduce((acc, area) => acc + area, 0)
  const rand = mulberry32(seed)

  const lines: string[] = []
  let nanCount = 0

  const channelMin = [Infinity, Infinity, Infinity]
  const channelMax = [-Infinity, -Infinity, -Infinity]
  const channelSum = [0, 0, 0]
  const perRectSum = new Map<number, number>()
  const perRectCount = new Map<number, number>()

  for (let n = 0; n < count; n++) {
    const rectIndex = sampleMode === "balanced"
      ? pickRectBalanced(n)
      : pickRect(rand, totalArea)
    const rect = scene.rects[rectIndex]
    const u = rand()
    const v = rand()

    const px = rect.corner.x + rect.edgeU.x * u + rect.edgeV.x * v
    const py = rect.corner.y + rect.edgeU.y * u + rect.edgeV.y * v
    const pz = rect.corner.z + rect.edgeU.z * u + rect.edgeV.z * v

    const indirect = traceIndirect(
      scene,
      px, py, pz,
      rect.normal.x, rect.normal.y, rect.normal.z,
      rect.albedo,
      spp,
      rand,
    )

    const rgb = [indirect.x, indirect.y, indirect.z]
    for (let c = 0; c < 3; c++) {
      if (!Number.isFinite(rgb[c])) nanCount++
      if (rgb[c] < channelMin[c]) channelMin[c] = rgb[c]
      if (rgb[c] > channelMax[c]) channelMax[c] = rgb[c]
      channelSum[c] += rgb[c]
    }

    const scalar = (rgb[0] + rgb[1] + rgb[2]) / 3
    perRectSum.set(rect.id, (perRectSum.get(rect.id) ?? 0) + scalar)
    perRectCount.set(rect.id, (perRectCount.get(rect.id) ?? 0) + 1)

    lines.push(JSON.stringify({
      p: [round(px), round(py), round(pz)],
      n: roundVec(rect.normal),
      albedo: roundVec(rect.albedo),
      indirect: rgb.map(round),
      rectId: rect.id,
    }))
  }

  const datasetPath = resolve(outDir, "indirect_dataset.jsonl")
  writeFileSync(datasetPath, lines.join("\n") + "\n", "utf8")

  console.log(`样本数：${count}，teacher spp：${spp}，seed：${seed}，采样模式：${sampleMode}`)
  console.log(`数据文件：${datasetPath}`)
  console.log(`NaN 计数：${nanCount}`)
  for (let c = 0; c < 3; c++) {
    const name = ["R", "G", "B"][c]
    console.log(`  ${name}  min=${round(channelMin[c])}  max=${round(channelMax[c])}  mean=${round(channelSum[c] / count)}`)
  }
  console.log("按面的平均间接光：")
  const rectIds = [...perRectCount.keys()].sort((a, b) => a - b)
  for (const id of rectIds) {
    const mean = (perRectSum.get(id) ?? 0) / (perRectCount.get(id) ?? 1)
    console.log(`  rect=${id}  mean=${round(mean)}`)
  }

  return datasetPath
}

function generatePixelProbe() {
  const aspect = pixelWidth / pixelHeight
  const halfHeight = Math.tan((65 * Math.PI) / 180 / 2)
  const camera = new PerspectiveCamera(65, aspect, 0.1, 100)
  camera.position.set(2.4, 2.2, 2.6)
  camera.lookAt(0, 0.9, 0)
  camera.updateMatrixWorld()

  const e = camera.matrixWorld.elements
  const position: [number, number, number] = [e[12], e[13], e[14]]
  const right: [number, number, number] = [e[0], e[1], e[2]]
  const up: [number, number, number] = [e[4], e[5], e[6]]
  const forward: [number, number, number] = [-e[8], -e[9], -e[10]]

  const pixels: unknown[] = []

  for (let j = 0; j < pixelHeight; j++) {
    for (let i = 0; i < pixelWidth; i++) {
      const rand = mulberry32((i * 1973 + j * 9277 + seed) >>> 0)

      let hitCount = 0
      let directR = 0
      let directG = 0
      let directB = 0
      let indirectR = 0
      let indirectG = 0
      let indirectB = 0
      let sampleP: [number, number, number] = [0, 0, 0]
      let sampleN: [number, number, number] = [0, 0, 0]
      let sampleAlbedo: [number, number, number] = [0, 0, 0]

      for (let s = 0; s < pixelSpp; s++) {
        const u = (i + rand()) / pixelWidth
        const v = (j + rand()) / pixelHeight
        const sx = (2 * u - 1) * aspect * halfHeight
        const sy = (1 - 2 * v) * halfHeight

        let dx = forward[0] + right[0] * sx + up[0] * sy
        let dy = forward[1] + right[1] * sx + up[1] * sy
        let dz = forward[2] + right[2] * sx + up[2] * sy
        const dl = Math.sqrt(dx * dx + dy * dy + dz * dz)
        dx /= dl
        dy /= dl
        dz /= dl

        const hit = intersectScene(scene, position[0], position[1], position[2], dx, dy, dz)
        if (hit.index < 0) continue
        if (hit.index === scene.lightIndex) continue

        const rect = scene.rects[hit.index]
        const hx = position[0] + dx * hit.t
        const hy = position[1] + dy * hit.t
        const hz = position[2] + dz * hit.t

        const D = sampleDirectLight(scene, hx, hy, hz, rect.normal.x, rect.normal.y, rect.normal.z, rect.albedo, rand)
        directR += D.x
        directG += D.y
        directB += D.z

        const I = traceIndirect(
          scene,
          hx, hy, hz,
          rect.normal.x, rect.normal.y, rect.normal.z,
          rect.albedo,
          1,
          rand,
        )
        indirectR += I.x
        indirectG += I.y
        indirectB += I.z

        sampleP = [hx, hy, hz]
        sampleN = [rect.normal.x, rect.normal.y, rect.normal.z]
        sampleAlbedo = [rect.albedo.x, rect.albedo.y, rect.albedo.z]
        hitCount++
      }

      const inv = hitCount > 0 ? 1 / hitCount : 0
      pixels.push({
        p: sampleP.map(round),
        n: sampleN.map(round),
        albedo: sampleAlbedo.map(round),
        direct: [round(directR * inv), round(directG * inv), round(directB * inv)],
        indirectGT: [round(indirectR * inv), round(indirectG * inv), round(indirectB * inv)],
        mask: hitCount > 0 ? 1 : 0,
      })
    }
  }

  const probePath = resolve(outDir, "pixel_probe.json")
  writeFileSync(
    probePath,
    JSON.stringify({
      width: pixelWidth,
      height: pixelHeight,
      samples: pixelSpp,
      camera: { position, forward, right, up, halfHeight, aspect },
      pixels,
    }),
    "utf8",
  )

  console.log(`像素探针：${pixelWidth}x${pixelHeight}，每像素 ${pixelSpp} 样本`)
  console.log(`探针文件：${probePath}`)
}

generateDataset()
if (withPixels) {
  generatePixelProbe()
}
