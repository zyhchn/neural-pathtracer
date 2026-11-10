import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

// 路由表：首页 + 两个演示页面（传统实现 / 神经网络实现）
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    // 传统路径追踪：WebGPU + TSL，多 bounce 的完整路径追踪
    {
      path: '/gpu-pathtrace',
      name: 'gpu-pathtrace',
      component: () => import('../views/gpu-pathtrace.vue'),
    },
    // 神经路径追踪：主光线 + NEE 直接光 + MLP 预测的间接光
    {
      path: '/neural-pathtrace',
      name: 'neural-pathtrace',
      component: () => import('../views/neural-pathtrace.vue'),
    },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

export default router
