"""训练「首个普通命中点的间接光」预测网络。

输入：X = [P(3), N(3), albedo(3)]，共 9 维
输出：indirect_rgb，共 3 维

标签由 training/generate-dataset.ts 用 CPU teacher 离线生成。
本脚本只负责学习真值，不做任何路径追踪。

用法（在项目根目录执行）：

    python training/train.py --data training/data/indirect_dataset.jsonl --out training/out
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="训练间接光预测 MLP")
    parser.add_argument("--data", default=os.path.join(here, "data", "indirect_dataset.jsonl"))
    parser.add_argument("--out", default=os.path.join(here, "out"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_dataset(path):
    """读取 JSONL，返回 X(9) / Y(3) / rectId"""
    xs = []
    ys = []
    rect_ids = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            xs.append(list(row["p"]) + list(row["n"]) + list(row["albedo"]))
            ys.append(list(row["indirect"]))
            rect_ids.append(int(row["rectId"]))
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(rect_ids, dtype=np.int64),
    )


class MLP(nn.Module):
    def __init__(self, in_dim=9, out_dim=3, hidden=128, layers=3):
        super().__init__()
        modules = []
        dim = in_dim
        for _ in range(layers):
            modules.append(nn.Linear(dim, hidden))
            modules.append(nn.SiLU())
            dim = hidden
        modules.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"设备：{device}")

    x_raw, y_raw, rect_ids = load_dataset(args.data)
    print(f"样本总数：{x_raw.shape[0]}，输入维度：{x_raw.shape[1]}，输出维度：{y_raw.shape[1]}")

    # 固定划分训练 / 验证
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(x_raw.shape[0])
    val_size = max(1, int(round(x_raw.shape[0] * args.val_ratio)))
    val_index = perm[:val_size]
    train_index = perm[val_size:]

    # 输入标准化
    x_mean = x_raw[train_index].mean(axis=0)
    x_std = x_raw[train_index].std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    # 目标先 log1p 再标准化，抑制动态范围
    y_log = np.log1p(y_raw)
    y_mean = y_log[train_index].mean(axis=0)
    y_std = y_log[train_index].std(axis=0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)

    x_norm = (x_raw - x_mean) / x_std
    y_norm = (y_log - y_mean) / y_std

    x_train = torch.tensor(x_norm[train_index], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_norm[train_index], dtype=torch.float32, device=device)
    x_val = torch.tensor(x_norm[val_index], dtype=torch.float32, device=device)
    y_val = torch.tensor(y_norm[val_index], dtype=torch.float32, device=device)

    model = MLP(in_dim=x_raw.shape[1], out_dim=y_raw.shape[1], hidden=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def denorm(y):
        return np.expm1(y * y_std + y_mean)

    y_val_raw = y_raw[val_index]
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, x_train.shape[0], args.batch):
            idx = order[start:start + args.batch]
            optimizer.zero_grad()
            pred = model(x_train[idx])
            loss = loss_fn(pred, y_train[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = float(loss_fn(val_pred, y_val).item())

        history["train"].append(epoch_loss / max(1, batches))
        history["val"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train={history['train'][-1]:.6f}  val={history['val'][-1]:.6f}")

        if patience_left <= 0:
            print(f"早停：连续 {args.patience} 个 epoch 验证集没有改善")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 在原始空间评估
    model.eval()
    with torch.no_grad():
        val_pred_norm = model(x_val).cpu().numpy()
    val_pred_raw = np.maximum(denorm(val_pred_norm), 0.0)

    diff = val_pred_raw - y_val_raw
    mae = np.abs(diff).mean(axis=0)
    rmse = np.sqrt((diff ** 2).mean(axis=0))
    rel = np.abs(diff).mean(axis=0) / np.maximum(y_val_raw.mean(axis=0), 1e-3)

    lines = []
    lines.append(f"设备：{device}")
    lines.append(f"样本总数：{x_raw.shape[0]}  训练：{train_index.shape[0]}  验证：{val_index.shape[0]}")
    lines.append(f"最优 epoch：{best_epoch}  最优验证 loss（归一化空间）：{best_val:.6f}")
    lines.append("")
    lines.append("验证集原始空间误差（每通道）：")
    for c, name in enumerate(["R", "G", "B"]):
        lines.append(f"  {name}  MAE={mae[c]:.6f}  RMSE={rmse[c]:.6f}  相对MAE={rel[c] * 100:.2f}%")
    lines.append(f"  平均  MAE={mae.mean():.6f}  RMSE={rmse.mean():.6f}  相对MAE={rel.mean() * 100:.2f}%")

    lines.append("")
    lines.append("验证集按面的平均绝对误差：")
    val_rect = rect_ids[val_index]
    for rid in sorted(set(val_rect.tolist())):
        subset = val_rect == rid
        lines.append(f"  rect={rid}  MAE={np.abs(diff[subset]).mean():.6f}  样本数={int(subset.sum())}")

    report = "\n".join(lines)
    print("")
    print(report)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": x_raw.shape[1],
            "out_dim": y_raw.shape[1],
            "hidden": args.hidden,
            "layers": args.layers,
        },
        os.path.join(args.out, "model.pt"),
    )
    np.savez(
        os.path.join(args.out, "norm.npz"),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        val_index=val_index,
        train_index=train_index,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle)
    with open(os.path.join(args.out, "metrics.txt"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")

    print("")
    print(f"模型：{os.path.join(args.out, 'model.pt')}")
    print(f"归一化参数：{os.path.join(args.out, 'norm.npz')}")
    print(f"训练曲线数据：{os.path.join(args.out, 'history.json')}")
    print(f"指标：{os.path.join(args.out, 'metrics.txt')}")


if __name__ == "__main__":
    main()
