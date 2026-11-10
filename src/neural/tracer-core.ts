// CPU 路径追踪核心：几何求交、NEE 直接光、余弦半球采样、间接光估计。
//
// 这份核心同时被两处使用，保证语义只有一份：
//   training/generate-dataset.ts  用 Node 跑，离线生成训练真值
//   src/neural/renderer.ts        在浏览器里跑神经渲染演示页
//
// 语义严格镜像 src/pathrace/app.ts 的 WebGPU/TSL 实现。
// 场景真源统一来自 src/pathrace/scene-room.ts。

import { buildRoomSceneData } from "../pathrace/scene-room.ts"
import type { RoomSceneData } from "../pathrace/scene-room.ts"

/** 与 app.ts 的 Loop({start: 0, end: 4}) 一致：主循环最多 4 个顶点 */
export const MAX_BOUNCES = 4
/** 与 app.ts 的 i >= 2 一致：从第 2 个顶点起做 Russian Roulette */
export const RR_START = 2
export const RR_MIN = 0.2
export const RR_MAX = 0.95
/** 与新射线起点偏移一致，防自交 */
export const BIAS = 1e-4
/** 与 app.ts 的 abs(denom) > 1e-6 一致 */
const DENOM_EPS = 1e-6
/** 与 app.ts 的 t > 1e-4 一致 */
const T_MIN = 1e-4

export type Vec3 = { x: number; y: number; z: number }

export type TeacherRect = {
  id: number
  materialId: number
  corner: Vec3
  edgeU: Vec3
  edgeV: Vec3
  normal: Vec3
  albedo: Vec3
  emission: Vec3
}

export type TeacherScene = {
  rects: TeacherRect[]
  /** 可采样训练点的矩形下标（已排除面光源） */
  sampleableIndices: number[]
  /** 与 sampleableIndices 一一对应的矩形面积，用于按面积加权采样 */
  sampleableAreas: number[]
  /** 面光源矩形下标 */
  lightIndex: number
  /** 面光源面积 */
  lightArea: number
}

/** 确定性随机源：同一 seed 必然产生同一序列，保证数据集与渲染可复现 */
export function mulberry32(seed: number) {
  let a = seed >>> 0
  return function (): number {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function length(v: Vec3) {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
}

function isEmissive(emission: Vec3) {
  return emission.x > 0 || emission.y > 0 || emission.z > 0
}

/** 从 scene-room.ts 的场景数据构建扁平场景 */
export function buildTeacherScene(sceneData: RoomSceneData): TeacherScene {
  const rects: TeacherRect[] = sceneData.rects.map((rect, index) => {
    const materialId = sceneData.materialIds[index]
    const material = sceneData.materials[materialId]
    return {
      id: rect.id,
      materialId,
      corner: { x: rect.corner.x, y: rect.corner.y, z: rect.corner.z },
      edgeU: { x: rect.edgeU.x, y: rect.edgeU.y, z: rect.edgeU.z },
      edgeV: { x: rect.edgeV.x, y: rect.edgeV.y, z: rect.edgeV.z },
      normal: { x: rect.normal.x, y: rect.normal.y, z: rect.normal.z },
      albedo: { x: material.albedo.x, y: material.albedo.y, z: material.albedo.z },
      emission: { x: material.emission.x, y: material.emission.y, z: material.emission.z },
    }
  })

  const lightIndex = rects.findIndex((rect) => rect.id === sceneData.lightRectId)
  if (lightIndex < 0) {
    throw new Error("tracer: 未能在场景中找到面光源")
  }

  const sampleableIndices: number[] = []
  const sampleableAreas: number[] = []
  for (let i = 0; i < rects.length; i++) {
    if (i === lightIndex) continue
    const area = length(rects[i].edgeU) * length(rects[i].edgeV)
    if (area <= 0) continue
    sampleableIndices.push(i)
    sampleableAreas.push(area)
  }

  const lightArea = length(rects[lightIndex].edgeU) * length(rects[lightIndex].edgeV)

  return { rects, sampleableIndices, sampleableAreas, lightIndex, lightArea }
}

/** 直接用 src/pathrace/scene-room.ts 的默认场景构建 */
export function buildDefaultScene(): TeacherScene {
  return buildTeacherScene(buildRoomSceneData())
}

/** 把所有非发光材质的反照率置零，用于自检「间接光是否收敛到 0」 */
export function blackenNonEmissiveAlbedo(scene: TeacherScene) {
  for (const rect of scene.rects) {
    if (isEmissive(rect.emission)) continue
    rect.albedo = { x: 0, y: 0, z: 0 }
  }
}

/** 判断某个矩形下标是否为面光源 */
export function isLight(scene: TeacherScene, index: number) {
  return index === scene.lightIndex
}

/**
 * 射线与场景求交，返回最近命中的下标与距离。
 * 与 app.ts 的 intersectSceneNode 一致：denom 保护 1e-6，t 保护 1e-4。
 * miss 时 index 为 -1。
 */
export function intersectScene(
  scene: TeacherScene,
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
): { t: number; index: number } {
  let bestT = Infinity
  let bestIndex = -1

  for (let i = 0; i < scene.rects.length; i++) {
    const rect = scene.rects[i]
    const n = rect.normal
    const denom = n.x * dx + n.y * dy + n.z * dz
    if (Math.abs(denom) < DENOM_EPS) continue

    const t = ((rect.corner.x - ox) * n.x + (rect.corner.y - oy) * n.y + (rect.corner.z - oz) * n.z) / denom
    if (!(t > T_MIN) || !(t < bestT)) continue

    const px = ox + dx * t
    const py = oy + dy * t
    const pz = oz + dz * t
    const relX = px - rect.corner.x
    const relY = py - rect.corner.y
    const relZ = pz - rect.corner.z

    const uu = rect.edgeU.x * rect.edgeU.x + rect.edgeU.y * rect.edgeU.y + rect.edgeU.z * rect.edgeU.z
    const vv = rect.edgeV.x * rect.edgeV.x + rect.edgeV.y * rect.edgeV.y + rect.edgeV.z * rect.edgeV.z
    const u = (relX * rect.edgeU.x + relY * rect.edgeU.y + relZ * rect.edgeU.z) / uu
    const v = (relX * rect.edgeV.x + relY * rect.edgeV.y + relZ * rect.edgeV.z) / vv
    if (u < 0 || u > 1 || v < 0 || v > 1) continue

    bestT = t
    bestIndex = i
  }

  return { t: bestT, index: bestIndex }
}

/**
 * 单样本 NEE：在面光源上均匀取点，做阴影射线，返回 RGB 直接光估计。
 * 与 app.ts 的 sampleDirectLightNode 一致。
 */
export function sampleDirectLight(
  scene: TeacherScene,
  px: number,
  py: number,
  pz: number,
  nx: number,
  ny: number,
  nz: number,
  albedo: Vec3,
  rand: () => number,
): Vec3 {
  const light = scene.rects[scene.lightIndex]
  const u = rand()
  const v = rand()

  const lx = light.corner.x + light.edgeU.x * u + light.edgeV.x * v
  const ly = light.corner.y + light.edgeU.y * u + light.edgeV.y * v
  const lz = light.corner.z + light.edgeU.z * u + light.edgeV.z * v

  const toX = lx - px
  const toY = ly - py
  const toZ = lz - pz
  const distSq = toX * toX + toY * toY + toZ * toZ
  const dist = Math.sqrt(distSq)
  if (dist <= 0) return { x: 0, y: 0, z: 0 }

  const wx = toX / dist
  const wy = toY / dist
  const wz = toZ / dist

  const cp = nx * wx + ny * wy + nz * wz
  const cl = -(light.normal.x * wx + light.normal.y * wy + light.normal.z * wz)
  if (!(cp > 0) || !(cl > 0)) return { x: 0, y: 0, z: 0 }

  const shadowHit = intersectScene(scene, px + nx * BIAS, py + ny * BIAS, pz + nz * BIAS, wx, wy, wz)
  if (shadowHit.index !== scene.lightIndex) return { x: 0, y: 0, z: 0 }

  const pdfArea = 1 / scene.lightArea
  const scale = (cp * cl) / (distSq * pdfArea)
  const k = scale / Math.PI

  return {
    x: albedo.x * light.emission.x * k,
    y: albedo.y * light.emission.y * k,
    z: albedo.z * light.emission.z * k,
  }
}

/**
 * 余弦加权半球采样，返回世界方向。
 * 与 app.ts 的 sampleCosineWorldNode 一致，包含同样的 up 轴选择规则。
 */
export function sampleCosineWorld(
  nx: number,
  ny: number,
  nz: number,
  rand: () => number,
): Vec3 {
  const u1 = rand()
  const u2 = rand()
  const r = Math.sqrt(u1)
  const phi = 2 * Math.PI * u2
  const x = r * Math.cos(phi)
  const y = r * Math.sin(phi)
  const z = Math.sqrt(1 - u1)

  let ux = 0
  let uy = 0
  let uz = 1
  if (Math.abs(nz) >= 0.999) {
    ux = 0
    uy = 1
    uz = 0
  }

  // tangent = normalize(cross(up, normal))
  let tx = uy * nz - uz * ny
  let ty = uz * nx - ux * nz
  let tz = ux * ny - uy * nx
  const tl = Math.sqrt(tx * tx + ty * ty + tz * tz)
  tx /= tl
  ty /= tl
  tz /= tl

  // bitangent = cross(normal, tangent)
  const bx = ny * tz - nz * ty
  const by = nz * tx - nx * tz
  const bz = nx * ty - ny * tx

  let rx = tx * x + bx * y + nx * z
  let ry = ty * x + by * y + ny * z
  let rz = tz * x + bz * y + nz * z
  const rl = Math.sqrt(rx * rx + ry * ry + rz * rz)

  return { x: rx / rl, y: ry / rl, z: rz / rl }
}

/**
 * 间接光 teacher：估计首个普通命中点的间接光。
 *
 * 语义：从该点出发先采一条余弦方向，然后统计「第 2 个顶点起」的所有 NEE 贡献。
 * 正好等于 app.ts 主循环中 i = 1, 2, 3 三轮累积的总和。
 *
 * @param spp 子路径采样数，越大标签噪声越小
 */
export function traceIndirect(
  scene: TeacherScene,
  px: number,
  py: number,
  pz: number,
  nx: number,
  ny: number,
  nz: number,
  albedo: Vec3,
  spp: number,
  rand: () => number,
): Vec3 {
  let sumR = 0
  let sumG = 0
  let sumB = 0

  for (let s = 0; s < spp; s++) {
    let br = albedo.x
    let bg = albedo.y
    let bb = albedo.z

    const wi = sampleCosineWorld(nx, ny, nz, rand)
    let ox = px + nx * BIAS
    let oy = py + ny * BIAS
    let oz = pz + nz * BIAS
    let dx = wi.x
    let dy = wi.y
    let dz = wi.z

    // MAX_BOUNCES 个顶点里，第 0 个顶点由调用方负责直接光，这里只跑剩下的 MAX_BOUNCES - 1 轮
    for (let k = 0; k < MAX_BOUNCES - 1; k++) {
      const hit = intersectScene(scene, ox, oy, oz, dx, dy, dz)
      if (hit.index < 0) break
      // 命中灯：不累加 emission，直接光已经由上一个顶点的 NEE 负责
      if (isLight(scene, hit.index)) break

      const rect = scene.rects[hit.index]
      const hx = ox + dx * hit.t
      const hy = oy + dy * hit.t
      const hz = oz + dz * hit.t

      const D = sampleDirectLight(scene, hx, hy, hz, rect.normal.x, rect.normal.y, rect.normal.z, rect.albedo, rand)
      sumR += br * D.x
      sumG += bg * D.y
      sumB += bb * D.z

      const wi2 = sampleCosineWorld(rect.normal.x, rect.normal.y, rect.normal.z, rand)
      br *= rect.albedo.x
      bg *= rect.albedo.y
      bb *= rect.albedo.z

      // 与 app.ts 一致：从第 2 个顶点起做 Russian Roulette
      if (k + 1 >= RR_START) {
        const q = Math.min(RR_MAX, Math.max(RR_MIN, Math.max(br, bg, bb)))
        if (rand() > q) break
        br /= q
        bg /= q
        bb /= q
      }

      ox = hx + rect.normal.x * BIAS
      oy = hy + rect.normal.y * BIAS
      oz = hz + rect.normal.z * BIAS
      dx = wi2.x
      dy = wi2.y
      dz = wi2.z
    }
  }

  const inv = 1 / spp
  return { x: sumR * inv, y: sumG * inv, z: sumB * inv }
}

/**
 * 传统路径追踪：直接光 + 多 bounce 间接光。
 * 用于在演示页里和神经版本做对照。
 */
export function traceRadiance(
  scene: TeacherScene,
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  rand: () => number,
): Vec3 {
  let radianceR = 0
  let radianceG = 0
  let radianceB = 0
  let br = 1
  let bg = 1
  let bb = 1

  let rox = ox
  let roy = oy
  let roz = oz
  let rdx = dx
  let rdy = dy
  let rdz = dz

  for (let i = 0; i < MAX_BOUNCES; i++) {
    const hit = intersectScene(scene, rox, roy, roz, rdx, rdy, rdz)
    if (hit.index < 0) break

    const rect = scene.rects[hit.index]
    const hx = rox + rdx * hit.t
    const hy = roy + rdy * hit.t
    const hz = roz + rdz * hit.t

    if (isLight(scene, hit.index)) {
      const facing = -(rect.normal.x * rdx + rect.normal.y * rdy + rect.normal.z * rdz) > 0
      if (facing && i === 0) {
        radianceR += rect.emission.x
        radianceG += rect.emission.y
        radianceB += rect.emission.z
      }
      break
    }

    const D = sampleDirectLight(scene, hx, hy, hz, rect.normal.x, rect.normal.y, rect.normal.z, rect.albedo, rand)
    radianceR += br * D.x
    radianceG += bg * D.y
    radianceB += bb * D.z

    const wi = sampleCosineWorld(rect.normal.x, rect.normal.y, rect.normal.z, rand)
    br *= rect.albedo.x
    bg *= rect.albedo.y
    bb *= rect.albedo.z

    if (i >= RR_START) {
      const q = Math.min(RR_MAX, Math.max(RR_MIN, Math.max(br, bg, bb)))
      if (rand() > q) break
      br /= q
      bg /= q
      bb /= q
    }

    rox = hx + rect.normal.x * BIAS
    roy = hy + rect.normal.y * BIAS
    roz = hz + rect.normal.z * BIAS
    rdx = wi.x
    rdy = wi.y
    rdz = wi.z
  }

  return { x: radianceR, y: radianceG, z: radianceB }
}
