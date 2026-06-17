import sys
import yaml
import torch
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from basicsr.models.archs.NAFNet_arch import NAFNet


def load_model(config_path, weights_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    net_opt = cfg["network_g"].copy()
    net_opt.pop("type", None)

    model = NAFNet(**net_opt)

    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            state = ckpt["params_ema"]
        elif "params" in ckpt:
            state = ckpt["params"]
        else:
            state = ckpt
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean_state[k] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()

    return model, cfg


def count_extra_skip_flops(cfg, img_size):
    net = cfg["network_g"]

    h = img_size
    w = img_size
    c = net.get("width", 32)
    img_channel = net.get("img_channel", 3)

    flops = 0

    for _ in net["enc_blk_nums"]:
        h //= 2
        w //= 2
        c *= 2

    for _ in net["dec_blk_nums"]:
        h *= 2
        w *= 2
        c //= 2

        # decoder skip connection: x = x + enc_skip
        flops += c * h * w

    # final residual: x = x + input
    flops += img_channel * img_size * img_size

    return flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="options/train/lowlight_nafnet_finetune_stage3_psnr.yml"
    )
    parser.add_argument(
        "--weights",
        default="experiments/lowlight_nafnet_finetune_stage3_psnr/models/net_g_final_best.pth"
    )
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    model, cfg = load_model(args.config, args.weights)

    params = sum(p.numel() for p in model.parameters())

    stats = defaultdict(float)

    def conv_hook(m, inp, out):
        x = inp[0]
        b = x.shape[0]
        out_c = out.shape[1]
        out_h = out.shape[2]
        out_w = out.shape[3]

        k_h, k_w = m.kernel_size
        in_c = m.in_channels
        groups = m.groups

        macs = b * out_c * out_h * out_w * (in_c // groups) * k_h * k_w
        stats["macs"] += macs

        if m.bias is not None:
            stats["element_flops"] += b * out_c * out_h * out_w

    def linear_hook(m, inp, out):
        x = inp[0]
        b = x.shape[0] if x.dim() > 1 else 1
        macs = b * m.in_features * m.out_features
        stats["macs"] += macs

        if m.bias is not None:
            stats["element_flops"] += b * m.out_features

    def pool_hook(m, inp, out):
        x = inp[0]
        stats["element_flops"] += x.numel()

    def layernorm_hook(m, inp, out):
        x = inp[0]
        # Approximation: mean, variance, normalize, scale, bias.
        stats["element_flops"] += 8 * x.numel()

    def simplegate_hook(m, inp, out):
        # SimpleGate: x1 * x2
        stats["element_flops"] += out.numel()

    def nafblock_hook(m, inp, out):
        x = inp[0]
        b, c, h, w = x.shape

        dw_channel = m.conv1.out_channels

        # SCA channel attention multiply
        stats["element_flops"] += b * (dw_channel // 2) * h * w

        # beta residual: inp + x * beta
        stats["element_flops"] += 2 * b * c * h * w

        # gamma residual: y + x * gamma
        stats["element_flops"] += 2 * b * c * h * w

    hooks = []

    for m in model.modules():
        name = m.__class__.__name__

        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, torch.nn.AdaptiveAvgPool2d):
            hooks.append(m.register_forward_hook(pool_hook))
        elif name == "LayerNorm2d":
            hooks.append(m.register_forward_hook(layernorm_hook))
        elif name == "SimpleGate":
            hooks.append(m.register_forward_hook(simplegate_hook))
        elif name == "NAFBlock":
            hooks.append(m.register_forward_hook(nafblock_hook))

    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    extra_skip_flops = count_extra_skip_flops(cfg, args.img_size)

    macs = stats["macs"]
    element_flops = stats["element_flops"] + extra_skip_flops

    # Convention:
    # 1 MAC = 1 multiplication + 1 addition = 2 FLOPs.
    total_flops = 2 * macs + element_flops

    print("========== FINAL MODEL SIZE / FLOPS ==========")
    print(f"Config: {args.config}")
    print(f"Weights: {args.weights}")
    print(f"Input size: 1 x 3 x {args.img_size} x {args.img_size}")
    print()
    print(f"Model size: {params / 1e6:.4f} M params")
    print(f"Conv/Linear MACs: {macs / 1e9:.4f} GMACs")
    print(f"Elementwise FLOPs: {element_flops / 1e9:.4f} GFLOPs")
    print(f"Total FLOPs: {total_flops / 1e9:.4f} GFLOPs")
    print()
    print("Final pipeline:")
    print("Input image -> NAFNet 43000 checkpoint -> raw PNG output")
    print("No OpenCV denoising or gamma correction is used in final metric submission.")
    print()
    print("Report to TA:")
    print(f"Size = {params / 1e6:.4f} M params")
    print(f"FLOPs = {total_flops / 1e9:.4f} GFLOPs for 256x256x3 input")
    print("Also mention convention: 1 MAC = 2 FLOPs.")
    print("==============================================")


if __name__ == "__main__":
    main()
