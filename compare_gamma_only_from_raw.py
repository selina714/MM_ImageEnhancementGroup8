import cv2
import numpy as np
from pathlib import Path

raw_dir = Path("visual_results/stage4_mse_compare_raw_vs_post/raw")
gt_dir = Path("/home/mislab/e/jonathan/tan/Datasets/nafnet_data/val/gt")

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

gammas = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]

scores = {g: [] for g in gammas}

raw_files = sorted(raw_dir.glob("*_raw.png"))

for raw_path in raw_files:
    stem = raw_path.stem.replace("_raw", "")
    gt_path = gt_dir / f"{stem}.webp"

    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

    if raw is None or gt is None:
        print("Skipping:", raw_path.name)
        continue

    for g in gammas:
        out = gamma_correct(raw, g)
        scores[g].append(psnr(out, gt))

print("========== Gamma-only PSNR Comparison ==========")
for g in gammas:
    print(f"raw + gamma {g:.2f}: {np.mean(scores[g]):.4f} dB")
print("Note: gamma 1.00 is the same as raw NAFNet output.")
print("================================================")
