<script setup lang="ts">
// 神经路径追踪演示页。
// 负责：加载 MLP 权重 -> 创建渲染器 -> 提供模式切换与状态显示。
import { onBeforeUnmount, onMounted, ref } from "vue"
import { NeuralPathTracer, type NeuralRenderMode } from "../neural/renderer"
import { loadMlpModel } from "../neural/mlp"

/** 挂载渲染器的容器 */
const container = ref<HTMLDivElement | null>(null)
/** 状态文案：加载进度 / 累积样本数 / 错误信息 */
const status = ref("正在加载模型…")
/** 致命错误信息，非空时替代 status 显示 */
const error = ref("")
/** 当前渲染模式 */
const mode = ref<NeuralRenderMode>("compare")

/** 工具栏上的三种模式按钮 */
const modes: Array<{ value: NeuralRenderMode; label: string; hint: string }> = [
  { value: "compare", label: "左右对比", hint: "左：神经间接光　右：传统路径追踪" },
  { value: "neural", label: "仅神经", hint: "主光线 + NEE 直接光 + MLP 预测间接光" },
  { value: "traditional", label: "仅传统", hint: "主光线 + NEE 直接光 + 多 bounce 间接光" },
]

/** 渲染器实例 */
let tracer: NeuralPathTracer | null = null

onMounted(async () => {
  const element = container.value
  if (!element) return

  try {
    // 模型 JSON 放在 public/models 下，随应用一起部署
    const modelUrl = `${import.meta.env.BASE_URL}models/indirect_mlp.json`
    const model = await loadMlpModel(modelUrl)

    // WebGPU 初始化是异步的，因此用静态工厂方法创建
    tracer = await NeuralPathTracer.create(element, model)
    tracer.onStatus = (text) => {
      status.value = text
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    status.value = ""
  }
})

onBeforeUnmount(() => {
  tracer?.dispose()
  tracer = null
})

/** 切换渲染模式 */
function setMode(next: NeuralRenderMode) {
  mode.value = next
  tracer?.setMode(next)
}
</script>

<template>
  <div class="page">
    <div class="stage">
      <div ref="container" class="canvas-host"></div>
    </div>

    <div class="hud">
      <div class="toolbar">
        <RouterLink class="back" to="/">← 返回目录</RouterLink>
        <button
          v-for="item in modes"
          :key="item.value"
          class="mode"
          :class="{ active: mode === item.value }"
          @click="setMode(item.value)"
        >
          {{ item.label }}
        </button>
      </div>

      <div class="info">
        <h1>神经间接光演示</h1>
        <p class="hint">{{ modes.find((item) => item.value === mode)?.hint }}</p>
        <p v-if="error" class="status error">{{ error }}</p>
        <p v-else class="status">{{ status }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  position: fixed;
  inset: 0;
  overflow: hidden;
}

.stage {
  position: absolute;
  inset: 0;
  background: #0b0f14;
}

.canvas-host {
  width: 100vw;
  height: 100vh;
}

.hud {
  position: absolute;
  top: 16px;
  left: 16px;
  right: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  pointer-events: none;
}

.toolbar,
.info {
  width: fit-content;
  max-width: min(100%, 920px);
  padding: 12px 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  background: rgba(11, 15, 20, 0.78);
  backdrop-filter: blur(10px);
  pointer-events: auto;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.back {
  display: inline-flex;
  align-items: center;
  padding: 7px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: var(--muted);
  font-size: 13px;
  text-decoration: none;
}

.back:hover {
  color: var(--accent);
  border-color: var(--accent);
}

.mode {
  padding: 7px 16px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: inherit;
  font-size: 13px;
  cursor: pointer;
}

.mode.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: var(--accent);
}

.info h1 {
  margin: 0 0 6px;
  font-size: 20px;
}

.hint {
  margin: 0;
  color: var(--muted);
  font-size: 12.5px;
}

.canvas-host :deep(canvas) {
  display: block;
  cursor: grab;
}

.status {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 12.5px;
}

.status.error {
  color: #ff8a8a;
}

@media (max-width: 720px) {
  .hud {
    top: 12px;
    left: 12px;
    right: 12px;
  }

  .toolbar,
  .info {
    width: auto;
  }
}
</style>
