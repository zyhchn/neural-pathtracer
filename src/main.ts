import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './styles.css'

// 应用入口：挂载根组件并装载路由。
createApp(App).use(router).mount('#app')
