<script setup lang="ts">
// 首页：介绍项目并给出两个演示页面的入口。
// demos 描述每个演示页面的标题、说明与路由地址。
const demos = [
  {
    to: '/neural-pathtrace',
    index: '01',
    name: '神经路径追踪',
    goal: '主光线 + NEE 直接光 + MLP 预测的间接光，可与传统结果左右对照',
  },
  {
    to: '/gpu-pathtrace',
    index: '02',
    name: '传统路径追踪',
    goal: 'WebGPU + TSL 实现的完整多 bounce 路径追踪，作为神经版本的对照基线',
  },
]
</script>

<template>
  <div class="home">
    <header class="home-header">
      <h1>神经路径追踪</h1>
      <p>用神经网络预测间接光，替代路径追踪中最昂贵的多次反弹计算。</p>
    </header>

    <nav class="catalog">
      <RouterLink
        v-for="demo in demos"
        :key="demo.to"
        class="catalog-item"
        :to="demo.to"
      >
        <span class="catalog-no">{{ demo.index }}</span>
        <span class="catalog-body">
          <strong>{{ demo.name }}</strong>
          <em>{{ demo.goal }}</em>
        </span>
        <span class="catalog-arrow">→</span>
      </RouterLink>
    </nav>
  </div>
</template>

<style scoped>
.home {
  max-width: 860px;
  margin: 0 auto;
  padding: 48px 24px 80px;
}

.home-header {
  margin: 0 0 28px;
}

.home-header h1 {
  margin: 0 0 8px;
  font-size: 24px;
}

.home-header p {
  margin: 0;
  color: var(--muted);
}

.catalog {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.catalog-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 14px 18px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
  color: inherit;
  text-decoration: none;
  transition: border-color 0.15s, background 0.15s;
}

.catalog-item:hover {
  border-color: var(--accent);
  background: var(--surface-hover);
}

.catalog-no {
  flex: none;
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
  font-size: 14px;
}

.catalog-body {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}

.catalog-body strong {
  font-size: 15px;
  font-weight: 600;
}

.catalog-body em {
  font-style: normal;
  font-size: 13px;
  color: var(--muted);
}

.catalog-arrow {
  margin-left: auto;
  color: var(--muted);
}
</style>
