// 传统路径追踪实现（WebGPU + TSL）
//
// 这是"传统"版本的参考实现：用 three.js 的 WebGPU 后端与 TSL 节点语言
// 在 GPU 上做完整的路径追踪（多个 bounce + NEE 直接光 + 俄罗斯轮盘赌），
// 结果累积到一张 storage texture 里，再通过一个全屏材质做色调映射输出。
//
// 相机与交互同样复用 three.js 的 PerspectiveCamera + OrbitControls。

import {
  PerspectiveCamera,
  Scene,
  OrthographicCamera,
  Mesh,
  PlaneGeometry,
  Vector3,
  MathUtils,
  HalfFloatType,
  NearestFilter,
} from "three"
import {
  NodeMaterial,
  StorageBufferAttribute,
  StorageBufferNode,
  StorageTexture,
  UniformNode,
  WebGPURenderer,
} from "three/webgpu"
import {
  abs,
  Break,
  cos,
  cross,
  float,
  Fn,
  hash,
  If,
  instanceIndex,
  int,
  ivec2,
  Loop,
  select,
  sin,
  storage,
  storageTexture,
  textureLoad,
  textureStore,
  uint,
  uniform,
  uv,
  vec3,
  vec4,
} from "three/tsl"
import { buildRoomSceneData, RoomSceneData } from "./scene-room"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"

/** 最大反弹次数，与训练真值 teacher 一致 */
const MAX_BOUNCES = 4
/** 曝光系数：线性 HDR -> 色调映射前的缩放 */
const EXPOSURE = 0.2

/** 创建用于观察场景的交互相机 */
function makeInteractCamera(width: number, height: number) {
  const camera = new PerspectiveCamera(65, width / height, 0.1, 100)
  camera.position.set(2.4, 2.2, 2.6)
  camera.lookAt(0, 0.9, 0)
  return camera
}

/** 从 three.js 相机解算出的渲染参数 */
export type CameraState = {
  origin: Vector3
  forward: Vector3
  right: Vector3
  up: Vector3
  halfHeight: number
  aspect: number
  width: number
  height: number
}

/**
 * 从相机矩阵提取渲染参数。
 * 矩阵列向量依次为 right / up / back，取反 back 得到 forward。
 */
export function readCameraState(
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

// ---- TSL 辅助节点：把常用计算封装成可复用的 Figma 节点函数 ----

/** 由像素坐标与帧号生成随机种子 */
const makePixelSeedNode: any = Fn((params: any) => {
  const px = params.px
  const py = params.py
  const frameIndex = params.frameIndex

  return px.mul(uint(1973))
    .add(py.mul(uint(9277)))
    .add(frameIndex.mul(uint(26699)))
}, { px: "uint", py: "uint", frameIndex: "uint", return: "uint" })

/** 由种子与维度取一个 [0,1) 浮点随机数 */
const randomFloatNode: any = Fn((params: any) => {
  const seed = params.seed
  const dim = params.dim

  return hash(seed.add(dim.mul(uint(1013904223))))
}, { seed: "uint", dim: "uint", return: "float" })

/** 生成主光线方向（带像素内抖动） */
const generatePrimaryRayDirNode: any = Fn((params: any) => {
  const px = params.px
  const py = params.py
  const width = params.width
  const height = params.height
  const seed = params.seed
  const forward = params.forward
  const right = params.right
  const up = params.up
  const aspect = params.aspect
  const halfHeight = params.halfHeight

  // 像素内均匀抖动，配合多帧累积实现抗锯齿
  const u = float(px)
    .add(randomFloatNode({ seed, dim: uint(0) }))
    .div(float(width))
    .toVar("u")
  const v = float(py)
    .add(randomFloatNode({ seed, dim: uint(1) }))
    .div(float(height))
    .toVar("v")

  const sx = u.mul(2.0).sub(1.0).mul(aspect).mul(halfHeight).toVar("sx")
  const sy = float(1.0).sub(v.mul(2.0)).mul(halfHeight).toVar("sy")

  return forward
    .add(right.mul(sx))
    .add(up.mul(sy))
    .normalize()
}, {
  px: "uint",
  py: "uint",
  width: "uint",
  height: "uint",
  seed: "uint",
  forward: "vec3",
  right: "vec3",
  up: "vec3",
  aspect: "float",
  halfHeight: "float",
  return: "vec3",
})

/** 俄罗斯轮盘赌的存活概率：由吞吐量决定，夹在 [0.2, 0.95] */
const computeContinueProbNode: any = Fn((params: any) => {
  const beta = params.beta
  return beta.r.max(beta.g).max(beta.b).max(0.2).min(0.95)
}, { beta: "vec3", return: "float" })

/** 余弦加权半球采样，返回世界方向 */
const sampleCosineWorldNode: any = Fn((params: any) => {
  const normal = params.normal
  const seed = params.seed

  const u1 = randomFloatNode({ seed, dim: uint(5) }).toVar("u1")
  const u2 = randomFloatNode({ seed, dim: uint(6) }).toVar("u2")

  const r = u1.sqrt().toVar("r")
  const phi = u2.mul(2.0 * Math.PI).toVar("phi")
  const x = r.mul(cos(phi)).toVar("x")
  const y = r.mul(sin(phi)).toVar("y")
  const z = float(1.0).sub(u1).sqrt().toVar("z")

  // 选一个与法线不平行的辅助轴构造切线空间
  const up = select(
    abs(normal.z).lessThan(0.999),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 0.0),
  ).toVar("up")

  const tangent = cross(up, normal).normalize().toVar("tangent")
  const bitangent = cross(normal, tangent).toVar("bitangent")

  return tangent.mul(x)
    .add(bitangent.mul(y))
    .add(normal.mul(z))
    .normalize()
}, { normal: "vec3", seed: "uint", return: "vec3" })

/**
 * 传统路径追踪应用。
 * 构造时完成所有 GPU 资源与管线搭建，并在内部启动渲染循环。
 */
export class PathTraceApp {

    /** 承载画布的容器 */
    container: HTMLElement
    scene: Scene
    camera: PerspectiveCamera
    /** 全屏显示用的正交相机 */
    viewCamera: OrthographicCamera
    renderer: WebGPURenderer

    /** 场景矩形 buffer（每个矩形 5 个 vec4） */
    rectBuffer!: StorageBufferNode<"vec4">
    /** 材质 buffer（每个材质 2 个 vec4） */
    materialBuffer!: StorageBufferNode<"vec4">
    /** 累积纹理：保存每像素的多次采样累加值 */
    accumTexture!: StorageTexture

    roomSceneData: RoomSceneData

    /** 当前相机参数 */
    cameraState!: CameraState

    // ---- 各类 uniform ----
    cameraUniform!: {
        origin: UniformNode<"vec3">
        forward: UniformNode<"vec3">
        right: UniformNode<"vec3">
        up: UniformNode<"vec3">
        halfHeight: UniformNode<"float", number>
        aspect: UniformNode<"float", number>
    }
    /** 面光源矩形下标 */
    lightIdUniform!: UniformNode<"uint", number>
    /** 场景矩形总数 */
    rectCountUniform!: UniformNode<"float", number>
    /** 材质总数 */
    materialCountUniform!: UniformNode<"float", number>
    /** 渲染宽度 */
    widthUniform!: UniformNode<"float", number>
    /** 渲染高度 */
    heightUniform!: UniformNode<"float", number>
    /** 已累积帧数 */
    frameIndexUniform!: UniformNode<"float", number>
    /** 路径追踪 compute 任务 */
    pathtraceCompute: any

    constructor(container: HTMLElement) {
        this.container = container

        const width = container.clientWidth
        const height = container.clientHeight
        this.scene = new Scene()
        this.camera = makeInteractCamera(width, height)

        this.viewCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1)

        // 显示材质：全屏平面，负责把累积纹理做色调映射后画出来
        const displayMaterial = new NodeMaterial()
        displayMaterial.fragmentNode = Fn(() => {
            const screenUV = uv().toVar("screenUV")
            const sampleX = int(screenUV.x.mul(this.widthUniform).min(this.widthUniform.sub(1.0))).toVar("sampleX")
            const sampleY = int(float(1.0).sub(screenUV.y).mul(this.heightUniform).min(this.heightUniform.sub(1.0))).toVar("sampleY")
            const coord = ivec2(sampleX, sampleY).toVar("coord")

            // 累加值除以帧数得到均值
            const sum = textureLoad(storageTexture(this.accumTexture).toReadOnly(), coord).rgb.toVar("sum")
            const sampleCount = float(this.frameIndexUniform).add(1.0).toVar("sampleCount")
            const mean = sum.div(sampleCount).toVar("mean")

            // 曝光 -> Reinhard 色调映射
            const exposed = mean.mul(EXPOSURE).toVar("exposed")
            const mapped = exposed.div(exposed.add(1.0)).toVar("mapped")

            return vec4(mapped, 1.0)
        })()
        this.scene.add(new Mesh(new PlaneGeometry(2, 2), displayMaterial))

        // WebGPU 渲染器
        this.renderer = new WebGPURenderer({
            antialias: true,
        })
        this.renderer.setSize(width, height)
        this.container.appendChild(this.renderer.domElement)
        this.renderer.init()

        // 轨道控制器：相机变化时重新读取参数并重置累积
        const controls = new OrbitControls(this.camera, this.renderer.domElement)
        controls.target.set(0, 0.9, 0)
        controls.update()

        controls.addEventListener("change", () => {
            this.cameraState = readCameraState(this.camera, width, height)
            this.updateCameraUniform()
            this.frameIndexUniform.value = 0
        })

        this.roomSceneData = buildRoomSceneData()

        this.cameraState = readCameraState(this.camera, width, height)

        this.createCameraUniform()

        this.createLightUniform()
        this.createRoomSceneBuffer()

        this.createRunUniform()

        /**
         * 场景求交：遍历所有矩形，返回最近命中。
         * 返回 vec3(found, hitIndex, t)，found 为 1/0。
         */
        const intersectSceneNode: any = Fn((params: any) => {
            const rayOrigin = params.rayOrigin
            const rayDir = params.rayDir
            const bestT = float(1e20).toVar("bestT")
            const hitFound = uint(0).toVar("hitFound")
            const sceneHitIndex = uint(0).toVar("sceneHitIndex")

            Loop({start: uint(0), end: uint(this.rectCountUniform), type: "uint", condition: "<"}, ({i}) => {
                const base = i.mul(5).toVar("base")

                const corner = this.rectBuffer.element(base.add(0)).xyz
                const edgeU = this.rectBuffer.element(base.add(1)).xyz
                const edgeV = this.rectBuffer.element(base.add(2)).xyz
                const normal = this.rectBuffer.element(base.add(3)).xyz

                const missT = float(1e20).toVar("missT")
                // 分母过小说明射线平行于平面
                const denom = normal.dot(rayDir).toVar("denom")

                If(abs(denom).greaterThan(1e-6), () => {
                    const t = corner.sub(rayOrigin).dot(normal).div(denom).toVar("t")

                    // t > 1e-4 用于防自交
                    If(t.greaterThan(1e-4), () => {
                        const p = rayOrigin.add(rayDir.mul(t)).toVar("p")
                        const rel = p.sub(corner).toVar("rel")

                        // 判断交点是否在矩形参数范围内
                        const u = rel.dot(edgeU).div(edgeU.dot(edgeU)).toVar("u")
                        const v = rel.dot(edgeV).div(edgeV.dot(edgeV)).toVar("v")

                        If(
                            u.greaterThanEqual(0.0).and(u.lessThanEqual(1.0))
                            .and(v.greaterThanEqual(0.0))
                            .and(v.lessThanEqual(1.0)),
                            () => {
                                missT.assign(t)
                            }
                        )
                    })
                })

                If(missT.lessThan(bestT), () => {
                    bestT.assign(missT)
                    hitFound.assign(uint(1))
                    sceneHitIndex.assign(i)
                })
            })
            return vec3(hitFound.toFloat(), sceneHitIndex.toFloat(), bestT)
        })

        /** 单样本 NEE：在面光源上取点做阴影射线，返回直接光 */
        const sampleDirectLightNode: any = Fn((params: any) => {
            const point = params.point
            const normal = params.normal
            const albedo = params.albedo
            const seed = params.seed

            const lightBase = this.lightIdUniform.mul(uint(5)).toVar("lightBase")
            const lightCorner = this.rectBuffer.element(lightBase.add(0)).xyz
            const lightEdgeU = this.rectBuffer.element(lightBase.add(1)).xyz
            const lightEdgeV = this.rectBuffer.element(lightBase.add(2)).xyz
            const lightNormal = this.rectBuffer.element(lightBase.add(3)).xyz
            const lightMaterialId = uint(this.rectBuffer.element(lightBase.add(4)).x).toVar("lightMaterialId")
            const lightEmission = this.materialBuffer.element(lightMaterialId.mul(2).add(1)).xyz

            // 在光源面上均匀取一点
            const lightU = randomFloatNode({ seed, dim: uint(2) }).toVar("lightU")
            const lightV = randomFloatNode({ seed, dim: uint(3) }).toVar("lightV")
            const lightPoint = lightCorner
                .add(lightEdgeU.mul(lightU))
                .add(lightEdgeV.mul(lightV))
                .toVar("lightPoint")

            const toLight = lightPoint.sub(point).toVar("toLight")
            const distanceSq = toLight.dot(toLight).toVar("distanceSq")
            const wi = toLight.normalize().toVar("wi")

            // 两侧余弦项，任一为负表示背对
            const cp = normal.dot(wi).max(0.0).toVar("cp")
            const cl = lightNormal.dot(wi).negate().max(0.0).toVar("cl")
            const direct = vec3(0.0).toVar("direct")

            If(cp.greaterThan(0.0).and(cl.greaterThan(0.0)), () => {
                // 阴影射线：必须能直接看到光源
                const shadowOrigin = point.add(normal.mul(1e-4)).toVar("shadowOrigin")
                const shadowHit = intersectSceneNode({
                    rayOrigin: shadowOrigin,
                    rayDir: wi,
                }).toVar("shadowHit")

                If(
                    shadowHit.x.greaterThan(0.5).and(uint(shadowHit.y).equal(this.lightIdUniform)),
                    () => {
                        // 面积采样 pdf 的倒数 = 光源面积；再乘 BRDF(1/π)
                        const lightArea = lightEdgeU.dot(lightEdgeU).sqrt().mul(lightEdgeV.dot(lightEdgeV).sqrt()).toVar("lightArea")
                        const pdfArea = float(1.0).div(lightArea).toVar("pdfArea")
                        const scale = cp.mul(cl).div(distanceSq.mul(pdfArea)).toVar("scale")
                        direct.assign(albedo.div(Math.PI).mul(lightEmission).mul(scale))
                    }
                )
            })

            return direct
        })

        /** 主路径追踪：逐像素采样一次完整路径 */
        const pathtraceNode = Fn(() => {
            const pixelIndex = uint(instanceIndex).toVar("pixelIndex")

            const width = uint(this.widthUniform).toVar("width")
            const height = uint(this.heightUniform).toVar("height")

            const px = pixelIndex.mod(width).toVar("px")
            const py = pixelIndex.div(width).toVar("py")

            const seed = makePixelSeedNode({
                px,
                py,
                frameIndex: uint(this.frameIndexUniform),
            }).toVar("seed")

            const rayOrigin = this.cameraUniform.origin.toVar("rayOrigin")

            const rayDir = generatePrimaryRayDirNode({
                px,
                py,
                width,
                height,
                seed,
                forward: this.cameraUniform.forward,
                right: this.cameraUniform.right,
                up: this.cameraUniform.up,
                aspect: this.cameraUniform.aspect,
                halfHeight: this.cameraUniform.halfHeight,
            }).toVar("rayDir")

            const radiance = vec3(0.0).toVar("radiance")
            // 路径吞吐量，随每次反弹乘上反照率
            const beta = vec3(1.0).toVar("beta")

            Loop({start: uint(0), end: uint(MAX_BOUNCES), type: "uint", condition: "<"}, ({i}) => {
                const bounceSeed = seed.add(i.mul(uint(1597334677))).toVar("bounceSeed")
                const hit = intersectSceneNode({
                    rayOrigin,
                    rayDir,
                }).toVar("hit")
                If(hit.x.lessThan(0.5), () => {
                    Break()
                })
                const bounceHitIndex = uint(hit.y).toVar("bounceHitIndex")
                const hitT = hit.z.toVar("hitT")
                const hitBase = bounceHitIndex.mul(5).toVar("hitBase")
                const hitPoint = rayOrigin.add(rayDir.mul(hitT)).toVar("hitPoint")
                const normal = this.rectBuffer.element(hitBase.add(3)).xyz
                const materialId = uint(this.rectBuffer.element(hitBase.add(4)).x)
                const id = uint(this.rectBuffer.element(hitBase.add(4)).y)
                const albedo = this.materialBuffer.element(materialId.mul(2)).xyz
                const emission = this.materialBuffer.element(materialId.mul(2).add(1)).xyz

                // 命中光源：只有主光线直接看到时计入自发光
                If(id.equal(this.lightIdUniform), () => {
                    const facing = normal.dot(rayDir).negate().greaterThan(0.0)
                    If(facing.and(i.equal(uint(0))), () => {
                        radiance.assign(radiance.add(emission))
                    })
                    Break()
                })

                // 当前顶点做一次 NEE
                const direct = sampleDirectLightNode({
                    point: hitPoint,
                    normal,
                    albedo,
                    seed: bounceSeed,
                }).toVar("direct")
                radiance.assign(radiance.add(beta.mul(direct)))

                // 继续采样下一段并更新吞吐量
                const wi = sampleCosineWorldNode({
                    normal,
                    seed: bounceSeed,
                }).toVar("wi")
                beta.assign(beta.mul(albedo))

                // 第 2 个 bounce 起做俄罗斯轮盘赌
                If(i.greaterThanEqual(uint(2)), () => {
                    const q = computeContinueProbNode({ beta }).toVar("q")
                    If(randomFloatNode({ seed: bounceSeed, dim: uint(4) }).greaterThan(q), () => {
                        Break()
                    })
                    beta.assign(beta.div(q))
                })

                rayOrigin.assign(hitPoint.add(normal.mul(1e-4)))
                rayDir.assign(wi)
            })

            // 累加进累积纹理：帧号为 0 时覆盖旧值
            const coord = ivec2(int(px), int(py)).toVar("coord")
            const prevSum = textureLoad(storageTexture(this.accumTexture).toReadWrite(), coord).rgb.toVar("prevSum")
            const baseSum = select(this.frameIndexUniform.lessThan(0.5), vec3(0.0, 0.0, 0.0), prevSum).toVar("baseSum")
            const nextSum = baseSum.add(radiance).toVar("nextSum")
            textureStore(storageTexture(this.accumTexture).toReadWrite(), coord, vec4(nextSum, 1.0))

            return nextSum
        })

        // 派发与像素数相同的线程数，每组 8x8
        this.pathtraceCompute = pathtraceNode().compute(width * height, [8, 8, 1])

        this.renderer.setAnimationLoop(() => this.tick())
    }

    /** 构建场景 buffer：矩形与材质各一份 */
    createRoomSceneBuffer() {
        const width = this.container.clientWidth
        const height = this.container.clientHeight
        const rectCount = this.roomSceneData.rects.length
        const materialCount = this.roomSceneData.materials.length
        const rectStride = 20

        const rectArray = new Float32Array(rectCount * 20)
        const materialArray = new Float32Array(materialCount * 8)

        // 每个矩形占 5 个 vec4：[corner][edgeU][edgeV][normal][materialId,rectId]
        for(let i = 0; i < rectCount; i++) {
            const rect = this.roomSceneData.rects[i]
            const materialId = this.roomSceneData.materialIds[i]
            const rectOffset = i * rectStride
            rectArray.set(rect.corner.toArray(), rectOffset)
            rectArray.set(rect.edgeU.toArray(), rectOffset + 4)
            rectArray.set(rect.edgeV.toArray(), rectOffset + 8)
            rectArray.set(rect.normal.toArray(), rectOffset + 12)
            rectArray.set([materialId, rect.id], rectOffset + 16)
        }

        // 每个材质占 2 个 vec4：[albedo][emission]
        for(let i = 0; i < materialCount; i++) {
            const material = this.roomSceneData.materials[i]
            const materialOffset = i * 8
            materialArray.set(material.albedo.toArray(), materialOffset)
            materialArray.set(material.emission.toArray(), materialOffset + 4)
        }

        const rectAttr = new StorageBufferAttribute(rectArray, 4)
        const materialAttr = new StorageBufferAttribute(materialArray, 4)

        const rectBuffer = storage(rectAttr, "vec4", rectArray.length/4).toReadOnly()
        const materialBuffer = storage(materialAttr, "vec4", materialArray.length/4).toReadOnly()

        this.rectBuffer = rectBuffer
        this.materialBuffer = materialBuffer

        // 累积纹理：半精度浮点，便于保存 HDR 累加值
        this.accumTexture = new StorageTexture(width, height)
        this.accumTexture.type = HalfFloatType
        this.accumTexture.minFilter = NearestFilter
        this.accumTexture.magFilter = NearestFilter
    }

    /** 创建相机相关 uniform */
    createCameraUniform() {
        const cameraOrigin = uniform(new Vector3())
        const cameraForward = uniform(new Vector3())
        const cameraRight = uniform(new Vector3())
        const cameraUp = uniform(new Vector3())
        const cameraHalfHeight = uniform(0.0)
        const cameraAspect = uniform(0.0)

        this.cameraUniform = {
            origin: cameraOrigin,
            forward: cameraForward,
            right: cameraRight,
            up: cameraUp,
            halfHeight: cameraHalfHeight,
            aspect: cameraAspect
        }
    }

    /** 把相机参数写入 uniform */
    updateCameraUniform() {
        this.cameraUniform.origin.value.copy(this.cameraState.origin)
        this.cameraUniform.forward.value.copy(this.cameraState.forward)
        this.cameraUniform.right.value.copy(this.cameraState.right)
        this.cameraUniform.up.value.copy(this.cameraState.up)
        this.cameraUniform.halfHeight.value = this.cameraState.halfHeight
        this.cameraUniform.aspect.value = this.cameraState.aspect
    }

    /** 面光源下标 uniform */
    createLightUniform() {
        const lightId = uniform(this.roomSceneData.lightRectId, "uint")
        this.lightIdUniform = lightId
    }

    /** 运行参数 uniform：数量、尺寸、帧号 */
    createRunUniform(){
        const rectCount = uniform(this.roomSceneData.rects.length)
        this.rectCountUniform = rectCount

        const materialCount = uniform(this.roomSceneData.materials.length)
        this.materialCountUniform = materialCount

        const width = uniform(this.container.clientWidth)
        const height = uniform(this.container.clientHeight)
        this.widthUniform = width
        this.heightUniform = height

        const frameIndex = uniform(0)
        this.frameIndexUniform = frameIndex
    }

    /** 每帧：跑一次 compute 采样并渲染显示 */
    tick() {
        this.updateCameraUniform()
        this.renderer.compute(this.pathtraceCompute)
        this.renderer.render(this.scene, this.viewCamera)
        this.frameIndexUniform.value++
    }
}
