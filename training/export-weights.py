"""把训练好的 MLP 权重导出成浏览器可直接加载的 JSON。

产物默认写到 public/models/indirect_mlp.json，供 /neural-pathtrace 页面 fetch 使用。

用法（在项目根目录执行）：

    python training/export-weights.py
"""

import argparse
import json
import os

import numpy as np
import torch


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    parser = argparse.ArgumentParser(description="导出 MLP 权重为 JSON")
    parser.add_argument("--model", default=os.path.join(here, "out", "model.pt"))
    parser.add_argument("--norm", default=os.path.join(here, "out", "norm.npz"))
    parser.add_argument("--out", default=os.path.join(root, "public", "models", "indirect_mlp.json"))
    parser.add_argument("--decimals", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]

    layers = []
    index = 0
    while f"net.{index}.weight" in state:
        weight = state[f"net.{index}.weight"].numpy().astype(np.float32)
        bias = state[f"net.{index}.bias"].numpy().astype(np.float32)
        layers.append({
            "inDim": int(weight.shape[1]),
            "outDim": int(weight.shape[0]),
            # 展平成行优先：[outDim][inDim]，索引 = out * inDim + i
            "w": np.round(weight.reshape(-1), args.decimals).tolist(),
            "b": np.round(bias, args.decimals).tolist(),
        })
        index += 2

    if not layers:
        raise RuntimeError(f"没有在 {args.model} 里找到 Linear 层，请检查 state_dict 结构")

    norm = np.load(args.norm)

    payload = {
        "inDim": int(checkpoint["in_dim"]),
        "outDim": int(checkpoint["out_dim"]),
        "hidden": int(checkpoint["hidden"]),
        "layers": int(checkpoint["layers"]),
        "xMean": np.round(norm["x_mean"], args.decimals).tolist(),
        "xStd": np.round(norm["x_std"], args.decimals).tolist(),
        "yMean": np.round(norm["y_mean"], args.decimals).tolist(),
        "yStd": np.round(norm["y_std"], args.decimals).tolist(),
        "weights": layers,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    size = os.path.getsize(args.out)
    print(f"输入维度：{payload['inDim']}  输出维度：{payload['outDim']}")
    print(f"线性层数：{len(layers)}  各层形状：{[(l['inDim'], l['outDim']) for l in layers]}")
    print(f"导出文件：{args.out}  ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
