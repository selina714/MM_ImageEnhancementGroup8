import cv2
import numpy as np
from pathlib import Path

h5_dir = Path("visual_results/stage4_mse_compare_raw_vs_post/denoise_h5")
gt_dir = Path("/home/mislab/e/jonathan/tan/Datasets/nafnet_data/val/gt")
out_dir = Path("visual_results/stage4_mse_compare_raw_vs_post/h5_gamma095")
out_dir.mkdir(parents=True, exist_ok=True)

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

gammas = [0.95, 0.98, 1.00, 1.02]
scores = {g: [] for g in gammas}

files = sorted(h5_dir.glob("*_h5.png"))

for f in files:
    stem = f.stem.replace("_h5", "")
    gt_path = gt_dir / f"{stem}.webp"

    h5 = cv2.imread(str(f), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

    if h5 is None or gt is None:
        print("Skipping:", f.name)
        continue

    for g in gammas:
        out = gamma_correct(h5, g)
        scores[g].append(psnr(out, gt))

        if abs(g - 0.95) < 1e-9:
            cv2.imwrite(str(out_dir / f"{stem}_h5_gamma095.png"), out)

print("========== Denoise h5 + Gamma PSNR Comparison ==========")
for g in gammas:
    print(f"h5 + gamma {g:.2f}: {np.mean(scores[g]):.4f} dB")
print("Note: gamma 1.00 is the same as denoise h5.")
print("========================================================")
