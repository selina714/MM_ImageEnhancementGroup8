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


def psnr(img, gt):
    img = img.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = np.mean((img - gt) ** 2)
    if mse == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def gamma_correct(img, gamma):
    table = np.array(
        [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(img, table)


def load_model(config_path, weight_path, device):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    opt = cfg["network_g"].copy()
    opt.pop("type", None)

    model = NAFNet(**opt).to(device)

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


def run_nafnet(model, img_bgr, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = torch.from_numpy(img_rgb).float()
    inp = inp.permute(2, 0, 1).unsqueeze(0) / 255.0
    inp = inp.to(device)

    with torch.no_grad():
        out = model(inp).clamp(0, 1)

    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="options/train/lowlight_nafnet_stage4_mse_lr5e6.yml")
    parser.add_argument("--weights", default="TA_submission/net_g_stage4_mse_best.pth")
    parser.add_argument("--val_lq", default="/home/mislab/e/jonathan/tan/Datasets/nafnet_data/val/lq")
    parser.add_argument("--val_gt", default="/home/mislab/e/jonathan/tan/Datasets/nafnet_data/val/gt")
    parser.add_argument("--output_dir", default="visual_results/stage4_mse_compare_raw_vs_post")
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.config, args.weights, device)

    lq_dir = Path(args.val_lq)
    gt_dir = Path(args.val_gt)
    out_dir = Path(args.output_dir)

    raw_dir = out_dir / "raw"
    h5_dir = out_dir / "denoise_h5"
    final_dir = out_dir / "h5_gamma080"
    compare_dir = out_dir / "compare"

    for d in [raw_dir, h5_dir, final_dir, compare_dir]:
        d.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in lq_dir.iterdir() if p.suffix.lower() in [".webp", ".png", ".jpg", ".jpeg"]])
    if args.max_images > 0:
        files = files[:args.max_images]

    scores = {
        "raw": [],
        "denoise_h5": [],
        "h5_gamma080": [],
    }

    print("Weights:", args.weights)
    print("Validation LQ:", args.val_lq)
    print("Validation GT:", args.val_gt)
    print("Images:", len(files))
    print("Output:", args.output_dir)
    print("Device:", device)

    for i, lq_path in enumerate(files):
        gt_path = gt_dir / lq_path.name

        img = cv2.imread(str(lq_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

        if img is None or gt is None:
            print("Skipping:", lq_path.name)
            continue

        raw = run_nafnet(model, img, device)
        h5 = cv2.fastNlMeansDenoisingColored(raw, None, 5, 5, 7, 21)
        h5_g080 = gamma_correct(h5, 0.80)

        scores["raw"].append(psnr(raw, gt))
        scores["denoise_h5"].append(psnr(h5, gt))
        scores["h5_gamma080"].append(psnr(h5_g080, gt))

        stem = lq_path.stem
        cv2.imwrite(str(raw_dir / f"{stem}_raw.png"), raw)
        cv2.imwrite(str(h5_dir / f"{stem}_h5.png"), h5)
        cv2.imwrite(str(final_dir / f"{stem}_h5_gamma080.png"), h5_g080)

        if i < 30:
            compare = np.concatenate([img, raw, h5, h5_g080, gt], axis=1)
            cv2.imwrite(str(compare_dir / f"{stem}_compare.png"), compare)

        print(f"[{i+1}/{len(files)}] {lq_path.name}")

    print("\n========== Validation PSNR Comparison ==========")
    for k, v in scores.items():
        print(f"{k}: {np.mean(v):.4f} dB")
    print("Higher PSNR is better.")
    print("Visual compare images saved in:", compare_dir)
    print("================================================")


if __name__ == "__main__":
    main()
