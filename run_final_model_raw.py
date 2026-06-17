import sys
import cv2
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from basicsr.models.archs.NAFNet_arch import NAFNet


def load_model(config_path, weight_path, device):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    net_opt = cfg["network_g"].copy()
    net_opt.pop("type", None)

    model = NAFNet(**net_opt).to(device)

    ckpt = torch.load(weight_path, map_location=device)

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

    return model


def find_input_files(input_dir):
    input_dir = Path(input_dir)

    files = sorted(input_dir.glob("*-in.webp"))
    if len(files) > 0:
        return files

    files = sorted(input_dir.glob("*.webp"))
    if len(files) > 0:
        return files

    all_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        all_files.extend(input_dir.glob(ext))

    return sorted(all_files)


def run_nafnet(model, img_bgr, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    inp = torch.from_numpy(img_rgb).float()
    inp = inp.permute(2, 0, 1).unsqueeze(0) / 255.0
    inp = inp.to(device)

    with torch.no_grad():
        out = model(inp)
        out = out.clamp(0, 1)

    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255.0).round().astype(np.uint8)

    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out_bgr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="options/train/lowlight_nafnet_finetune_stage3_psnr.yml"
    )
    parser.add_argument(
        "--weights",
        default="experiments/lowlight_nafnet_finetune_stage3_psnr/models/net_g_43000.pth"
    )
    parser.add_argument(
        "--input_dir",
        default="datasets/lowlight/test"
    )
    parser.add_argument(
        "--output_dir",
        default="TA_submission/nafnet_raw"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.config, args.weights, device)

    files = find_input_files(args.input_dir)

    if args.max_images > 0:
        files = files[:args.max_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Final inference pipeline: NAFNet only")
    print("Device:", device)
    print("Weights:", args.weights)
    print("Input dir:", args.input_dir)
    print("Output dir:", output_dir)
    print("Images:", len(files))

    for i, img_path in enumerate(files):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            print("Cannot read:", img_path)
            continue

        pred = run_nafnet(model, img, device)

        # Convert input name to TA output name.
        # Example: 123-in.webp -> 123.png
        stem = img_path.stem.replace("-in", "")
        save_path = output_dir / f"{stem}.png"

        cv2.imwrite(str(save_path), pred)

        print(f"[{i + 1}/{len(files)}] saved {save_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
