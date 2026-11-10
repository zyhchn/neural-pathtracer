import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 构建配置：启用 Vue 单文件组件支持，并固定开发端口。
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
  },
})
