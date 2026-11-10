// 场景定义：Cornell Box 风格的封闭房间 + 一个中央立方体。
//
// 场景完全由「矩形」组成，每个矩形用参数式表达：
//   point(u, v) = corner + edgeU * u + edgeV * v ，其中 u、v ∈ [0, 1]
// 这样求交只需解一个平面方程再判断参数范围，CPU teacher 与 GPU shader 共用同一套数据。
//
// 房间六面 + 面光源 + 立方体六面，共 13 个矩形；
// 左墙红色、右墙绿色，用于观察间接光在墙壁间的颜色反弹（color bleeding）。

import { Vector3 } from "three"

/** 一个矩形面：由角点和两条边向量张成，normal 决定朝向 */
export type RoomRect = {
  id: number
  name: string
  corner: Vector3
  edgeU: Vector3
  edgeV: Vector3
  normal: Vector3
}

/** 材质：漫反射反照率 + 自发光（面光源把 emission 设为光源亮度） */
export type RGBMaterial = {
  albedo: Vector3
  emission: Vector3
}

/** 完整场景数据：矩形列表、每个矩形用到的材质下标、材质列表、光源矩形下标 */
export type RoomSceneData = {
  rects: RoomRect[]
  materialIds: number[]
  materials: RGBMaterial[]
  lightRectId: number
}

// 材质下标（materials 数组里的位置）
export const ROOM_NEUTRAL_MATERIAL = 0
export const ROOM_LEFT_RED_MATERIAL = 1
export const ROOM_RIGHT_GREEN_MATERIAL = 2
export const LIGHT_MATERIAL = 3
export const CUBE_MATERIAL = 4

// 各材质的反照率 / 自发光参数
/** 中性墙面：略偏暖的灰白 */
export const ROOM_NEUTRAL_ALBEDO = new Vector3(0.75, 0.75, 0.72)
/** 左墙：红色 */
export const ROOM_LEFT_RED_ALBEDO = new Vector3(0.75, 0.18, 0.12)
/** 右墙：绿色 */
export const ROOM_RIGHT_GREEN_ALBEDO = new Vector3(0.18, 0.65, 0.2)
/** 立方体：浅灰 */
export const CUBE_ALBEDO = new Vector3(0.68, 0.68, 0.72)
/** 面光源亮度 */
export const LIGHT_EMISSION = new Vector3(12, 12, 12)

/** 面光源矩形的 id（同时被 shader 当作光源下标使用） */
export const LIGHT_RECT_ID = 6

/** 构造全部矩形：房间六面（法线朝内）+ 面光源 + 立方体六面（法线朝外） */
function createRoomRects(): RoomRect[] {
  return [
    // 房间六面，法线朝内
    {
      id: 0,
      name: "floor",
      corner: new Vector3(-3, 0, -3),
      edgeU: new Vector3(6, 0, 0),
      edgeV: new Vector3(0, 0, 6),
      normal: new Vector3(0, 1, 0),
    },
    {
      id: 1,
      name: "right-wall",
      corner: new Vector3(3, 0, -3),
      edgeU: new Vector3(0, 4, 0),
      edgeV: new Vector3(0, 0, 6),
      normal: new Vector3(-1, 0, 0),
    },
    {
      id: 2,
      name: "left-wall",
      corner: new Vector3(-3, 0, -3),
      edgeU: new Vector3(0, 4, 0),
      edgeV: new Vector3(0, 0, 6),
      normal: new Vector3(1, 0, 0),
    },
    {
      id: 3,
      name: "back-wall",
      corner: new Vector3(-3, 0, -3),
      edgeU: new Vector3(6, 0, 0),
      edgeV: new Vector3(0, 4, 0),
      normal: new Vector3(0, 0, 1),
    },
    {
      id: 4,
      name: "front-wall",
      corner: new Vector3(-3, 0, 3),
      edgeU: new Vector3(6, 0, 0),
      edgeV: new Vector3(0, 4, 0),
      normal: new Vector3(0, 0, -1),
    },
    {
      id: 5,
      name: "ceiling",
      corner: new Vector3(-3, 4, -3),
      edgeU: new Vector3(6, 0, 0),
      edgeV: new Vector3(0, 0, 6),
      normal: new Vector3(0, -1, 0),
    },
    // 面光源
    {
      id: LIGHT_RECT_ID,
      name: "light",
      corner: new Vector3(-2.4, 3.98, -1),
      edgeU: new Vector3(1.2, 0, 0),
      edgeV: new Vector3(0, 0, 2),
      normal: new Vector3(0, -1, 0),
    },
    // 中央立方体六面，法线朝外
    {
      id: 7,
      name: "cube-pos-x",
      corner: new Vector3(0.65, 0, -0.65),
      edgeU: new Vector3(0, 1.3, 0),
      edgeV: new Vector3(0, 0, 1.3),
      normal: new Vector3(1, 0, 0),
    },
    {
      id: 8,
      name: "cube-neg-x",
      corner: new Vector3(-0.65, 0, -0.65),
      edgeU: new Vector3(0, 1.3, 0),
      edgeV: new Vector3(0, 0, 1.3),
      normal: new Vector3(-1, 0, 0),
    },
    {
      id: 9,
      name: "cube-pos-y",
      corner: new Vector3(-0.65, 1.3, -0.65),
      edgeU: new Vector3(1.3, 0, 0),
      edgeV: new Vector3(0, 0, 1.3),
      normal: new Vector3(0, 1, 0),
    },
    {
      id: 10,
      name: "cube-neg-y",
      corner: new Vector3(-0.65, 0, -0.65),
      edgeU: new Vector3(1.3, 0, 0),
      edgeV: new Vector3(0, 0, 1.3),
      normal: new Vector3(0, -1, 0),
    },
    {
      id: 11,
      name: "cube-pos-z",
      corner: new Vector3(-0.65, 0, 0.65),
      edgeU: new Vector3(1.3, 0, 0),
      edgeV: new Vector3(0, 1.3, 0),
      normal: new Vector3(0, 0, 1),
    },
    {
      id: 12,
      name: "cube-neg-z",
      corner: new Vector3(-0.65, 0, -0.65),
      edgeU: new Vector3(1.3, 0, 0),
      edgeV: new Vector3(0, 1.3, 0),
      normal: new Vector3(0, 0, -1),
    },
  ]
}

/**
 * 组装完整场景：矩形列表 + 材质列表 + 每个矩形对应的材质下标。
 * 材质下标决定墙面颜色：右墙绿、左墙红、立方体浅灰、其余中性。
 */
export function buildRoomSceneData(): RoomSceneData {
  const rects = createRoomRects()

  const materials: RGBMaterial[] = [
    { albedo: ROOM_NEUTRAL_ALBEDO.clone(), emission: new Vector3(0, 0, 0) },
    { albedo: ROOM_LEFT_RED_ALBEDO.clone(), emission: new Vector3(0, 0, 0) },
    { albedo: ROOM_RIGHT_GREEN_ALBEDO.clone(), emission: new Vector3(0, 0, 0) },
    { albedo: new Vector3(0, 0, 0), emission: LIGHT_EMISSION.clone() },
    { albedo: CUBE_ALBEDO.clone(), emission: new Vector3(0, 0, 0) },
  ]

  const materialIds = rects.map((rect) => {
    if (rect.id === LIGHT_RECT_ID) return LIGHT_MATERIAL
    if (rect.id >= 7) return CUBE_MATERIAL
    if (rect.id === 1) return ROOM_RIGHT_GREEN_MATERIAL
    if (rect.id === 2) return ROOM_LEFT_RED_MATERIAL
    return ROOM_NEUTRAL_MATERIAL
  })

  return {
    rects,
    materialIds,
    materials,
    lightRectId: LIGHT_RECT_ID,
  }
}
