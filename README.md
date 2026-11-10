# 神经路径追踪（Neural Path Tracer）

用神经网络预测**间接光**，替代路径追踪中最昂贵的「多次反弹」计算。

本项目同时提供两套实现，便于对照：

| 页面 | 实现 | 说明 |
| --- | --- | --- |
| `/neural-pathtrace` | 原生 WebGPU + WGSL | 主光线 + NEE 直接光 + **MLP 预测的间接光**，可与传统结果左右对照 |
| `/gpu-pathtrace` | three.js WebGPU 后端 + TSL | 完整的多 bounce 路径追踪，作为神经版本的基线 |

核心思路：在一个 Cornell Box 风格场景中，神经网络只学习「命中点 → 间接光 RGB」这段映射，
于是每条光线只需一次求交即可得到间接光，省去后续所有反弹采样。

## 环境要求

- Node.js ≥ 22.6（训练数据生成脚本直接以 TS 运行，依赖 Node 的类型擦除能力）
- 浏览器需支持 WebGPU（Chrome / Edge 113+）
- Python 3.10+（仅训练与可视化需要）

## 快速开始（前端）

```bash
npm install
npm run dev        # 开发服务器，默认 http://localhost:5173
npm run build      # 类型检查 + 生产构建
npm run preview    # 预览构建产物
```

打开首页后进入 `神经路径追踪` 页面即可看到左右对照效果（左：神经，右：传统）。
鼠标拖拽可旋转视角，切换视角后累积会重置并重新收敛。

## 目录结构

```
src/
  neural/
    renderer.ts      原生 WebGPU 神经渲染器（compute + render 两段 shader）
    mlp.ts           浏览器端 MLP 推理（加载 JSON 权重）
    tracer-core.ts   CPU 路径追踪核心（几何求交 / NEE / 间接光真值）
  pathrace/
    app.ts           传统路径追踪（WebGPU + TSL）
    scene-room.ts    场景定义（矩形 + 材质），前端与训练脚本共用
  views/             两个演示页面
training/
  generate-dataset.ts  采样训练点，用 CPU teacher 生成真值数据集
  train.py             训练 MLP
  export-weights.py    把权重导出成浏览器可加载的 JSON
  visualize.py         训练结果可视化与 A/B 对照图
  data/                数据集与像素探针
  out/                 模型、归一化参数、指标与图表
public/models/
  indirect_mlp.json    前端实际加载的模型权重
```

## 训练流程

1. **生成数据集**（在项目根目录执行）

   ```bash
   npm run gen:data -- --count 8000 --spp 512 --sample-mode balanced --with-pixels
   ```

   - `--count` 训练点数量，`--spp` 每个点的子路径采样数（越大标签噪声越小）
   - `--sample-mode balanced` 按面均衡采样，保证小物体也拿到足够监督
   - `--with-pixels` 额外生成 `training/data/pixel_probe.json`，用于出 A/B 对照图
   - 产物：`training/data/indirect_dataset.jsonl`

2. **训练**

   ```bash
   python training/train.py --data training/data/indirect_dataset.jsonl --out training/out --hidden 64
   ```

   - 输入 9 维 `[位置(3), 法线(3), 反照率(3)]`，输出 3 维 `indirect_rgb`
   - 目标先 `log1p` 再标准化，抑制动态范围；带早停
   - 产物：`model.pt`、`norm.npz`、`history.json`、`metrics.txt`

3. **导出权重**

   ```bash
   python training/export-weights.py
   ```

   产物写到 `public/models/indirect_mlp.json`，供前端 `fetch` 加载。

4. **可视化（可选）**

   ```bash
   python training/visualize.py --probe training/data/pixel_probe.json
   ```

   在 `training/out/` 生成 loss 曲线、散点图、误差分布与 A/B 对照图。

> 仓库内已附带一份训练好的模型与图表：`training/out/` 与 `public/models/indirect_mlp.json`。

## 模型结构

```
输入 9 维 → Linear(9→64) → SiLU → Linear(64→64) → SiLU → Linear(64→64) → SiLU → Linear(64→3)
```

- 训练在标准化空间进行，推理时先做输入标准化、再做输出反标准化 + `expm1`
- 隐藏层宽度固定为 **64**，与 GPU 渲染器里 WGSL 的固定权重布局一一对应；
  若改动网络结构，需同步修改 `src/neural/renderer.ts` 中的布局常量与校验
- 仓库自带模型指标（验证集）：平均 MAE ≈ 0.0087，相对 MAE ≈ 7.4%
  （详见 `training/out/metrics.txt`）

## 实现要点

- **两段 shader**：`compute` 负责逐像素路径追踪 / MLP 推理并累加到 storage buffer；
  `render` 负责取值、按帧数求均值、做曝光 → Reinhard → sRGB 后输出到画布。
  两段 shader 共用布局但 `accum` 的读写类型不同（`read_write` vs `read`），必须分开声明。
- **相机复用**：相机与交互控制仍用 three.js 的 `PerspectiveCamera` + `OrbitControls`，
  仅作为参数来源，渲染完全交给原生 WebGPU。
- **渐进式累积**：每帧对每像素累加一个新样本，画面随时间收敛；视角变化时重置累积。
