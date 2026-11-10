"""可视化训练结果，并生成预测 / 真值对照图。

产物默认写到 training/out/：
    loss.png            训练与验证 loss 曲线
    scatter.png         预测 vs 真值散点（RGB 三个子图）
    error_hist.png      相对误差分布
    error_by_rect.png   按面的平均绝对误差
    ab_reference.png    参考图：direct + 真值间接光
    ab_predicted.png    预测图：direct + 预测间接光
    ab_error.png        两者差异放大图
    ab_compare.png      三者横向拼接

用法（在项目根目录执行）：

    python training/visualize.py --probe training/data/pixel_probe.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="可视化间接光预测结果")
    parser.add_argument("--data", default=os.path.join(here, "data", "indirect_dataset.jsonl"))
    parser.add_argument("--model", default=os.path.join(here, "out", "model.pt"))
    parser.add_argument("--norm", default=os.path.join(here, "out", "norm.npz"))
    parser.add_argument("--history", default=os.path.join(here, "out", "history.json"))
    parser.add_argument("--probe", default=os.path.join(here, "data", "pixel_probe.json"))
    parser.add_argument("--out", default=os.path.join(here, "out"))
    parser.add_argument("--exposure", type=float, default=0.2)
    parser.add_argument("--error-gain", type=float, default=4.0)
    return parser.parse_args()


def load_dataset(path):
    xs, ys, rect_ids = [], [], []
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


def load_model(model_path, norm_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MLP(
        in_dim=checkpoint["in_dim"],
        out_dim=checkpoint["out_dim"],
        hidden=checkpoint["hidden"],
        layers=checkpoint["layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    norm = np.load(norm_path)
    return model, norm


def predict_raw(model, norm, x_raw):
    """输入原始 9 维特征，输出原始空间的 indirect_rgb"""
    x_norm = (x_raw - norm["x_mean"]) / norm["x_std"]
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm, dtype=torch.float32)).numpy()
    y_log = y_norm * norm["y_std"] + norm["y_mean"]
    return np.maximum(np.expm1(y_log), 0.0)


def present(linear, exposure):
    """与项目其余部分一致：曝光 -> Reinhard -> sRGB 编码"""
    exposed = np.maximum(linear, 0.0) * exposure
    mapped = exposed / (1.0 + exposed)
    return np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2) * 255.0


def plot_loss(args):
    history_path = args.history
    if not os.path.exists(history_path):
        print(f"跳过 loss 曲线：找不到 {history_path}")
        return
    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(history["train"], label="train", color="#3b82f6")
    ax.plot(history["val"], label="val", color="#ef4444")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized space)")
    ax.set_yscale("log")
    ax.set_title("loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "loss.png"))
    plt.close(fig)
    print("已写出 loss.png")


def plot_scatter(y_true, y_pred, args):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    limit = float(max(y_true.max(), y_pred.max(), 1e-6))
    for c, name in enumerate(["R", "G", "B"]):
        ax = axes[c]
        ax.scatter(y_true[:, c], y_pred[:, c], s=4, alpha=0.25, color=["#ef4444", "#22c55e", "#3b82f6"][c])
        ax.plot([0, limit], [0, limit], color="#111827", linewidth=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("ground truth")
        ax.set_ylabel("prediction")
        ax.set_title(f"{name} channel")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "scatter.png"))
    plt.close(fig)
    print("已写出 scatter.png")


def plot_error_hist(y_true, y_pred, args):
    eps = 1e-3
    rel = np.abs(y_pred - y_true) / np.maximum(y_true, eps)
    rel = rel.reshape(-1)
    rel = rel[np.isfinite(rel)]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.hist(np.clip(rel, 0, 2), bins=80, color="#6366f1")
    ax.set_xlabel("|pred - gt| / (gt + 1e-3)")
    ax.set_ylabel("count")
    ax.set_title(f"relative error   mean={rel.mean() * 100:.2f}%   median={np.median(rel) * 100:.2f}%")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_hist.png"))
    plt.close(fig)
    print("已写出 error_hist.png")


def plot_error_by_rect(y_true, y_pred, rect_ids, args):
    ids = sorted(set(rect_ids.tolist()))
    means = [np.abs(y_pred[rect_ids == rid] - y_true[rect_ids == rid]).mean() for rid in ids]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.bar([str(rid) for rid in ids], means, color="#0ea5e9")
    ax.set_xlabel("rect id")
    ax.set_ylabel("MAE")
    ax.set_title("mean absolute error by rect")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "error_by_rect.png"))
    plt.close(fig)
    print("已写出 error_by_rect.png")


def render_ab(args, model, norm):
    if not os.path.exists(args.probe):
        print(f"跳过 A/B 对照图：找不到 {args.probe}")
        return

    with open(args.probe, "r", encoding="utf-8") as handle:
        probe = json.load(handle)

    width = probe["width"]
    height = probe["height"]
    pixels = probe["pixels"]

    p = np.asarray([px["p"] for px in pixels], dtype=np.float64)
    n = np.asarray([px["n"] for px in pixels], dtype=np.float64)
    albedo = np.asarray([px["albedo"] for px in pixels], dtype=np.float64)
    direct = np.asarray([px["direct"] for px in pixels], dtype=np.float64)
    indirect_gt = np.asarray([px["indirectGT"] for px in pixels], dtype=np.float64)
    mask = np.asarray([px["mask"] for px in pixels], dtype=np.float64)

    indirect_pred = predict_raw(model, norm, np.concatenate([p, n, albedo], axis=1))

    reference_linear = (direct + indirect_gt) * mask[:, None]
    predicted_linear = (direct + indirect_pred) * mask[:, None]
    error_linear = np.abs(predicted_linear - reference_linear)

    def to_image(linear, gain=1.0):
        img = present(linear * gain, args.exposure)
        return img.reshape(height, width, 3).astype(np.uint8)

    reference_img = to_image(reference_linear)
    predicted_img = to_image(predicted_linear)
    error_img = to_image(error_linear, gain=args.error_gain)

    mpimg.imsave(os.path.join(args.out, "ab_reference.png"), reference_img)
    mpimg.imsave(os.path.join(args.out, "ab_predicted.png"), predicted_img)
    mpimg.imsave(os.path.join(args.out, "ab_error.png"), error_img)
    mpimg.imsave(
        os.path.join(args.out, "ab_compare.png"),
        np.concatenate([reference_img, predicted_img, error_img], axis=1),
    )

    mae = error_linear[mask > 0].mean() if np.any(mask > 0) else float("nan")
    print(f"已写出 ab_reference.png / ab_predicted.png / ab_error.png / ab_compare.png")
    print(f"探针像素平均绝对误差（线性空间）：{mae:.6f}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    x_raw, y_raw, rect_ids = load_dataset(args.data)
    model, norm = load_model(args.model, args.norm)

    val_index = norm["val_index"]
    x_val = x_raw[val_index]
    y_val = y_raw[val_index]
    rect_val = rect_ids[val_index]
    y_pred = predict_raw(model, norm, x_val)

    plot_loss(args)
    plot_scatter(y_val, y_pred, args)
    plot_error_hist(y_val, y_pred, args)
    plot_error_by_rect(y_val, y_pred, rect_val, args)
    render_ab(args, model, norm)

    print("")
    print(f"全部图已写入：{args.out}")


if __name__ == "__main__":
    main()
