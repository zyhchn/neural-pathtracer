<script setup lang="ts">
// 传统路径追踪演示页。
// 挂载后实例化 PathTraceApp，由它接管容器并启动 WebGPU 渲染循环。
import { PathTraceApp } from "../pathrace/app"
import { onMounted } from "vue"

onMounted(() => {
  // 渲染器会在构造时创建画布并插入该容器
  const app = new PathTraceApp(document.getElementById("container") as HTMLElement)
})
</script>

<template>
  <!-- 全屏容器，路径追踪画面渲染在这里 -->
  <div id="container">
  </div>
</template>

<style scoped>
#container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}
</style>
