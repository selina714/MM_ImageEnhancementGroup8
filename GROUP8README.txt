# MM Image Enhancement Group 8

This project performs low-light image enhancement using a fine-tuned NAFNet model.
Our final pipeline is:

```text
Input low-light image → Stage 4 MSE fine-tuned NAFNet → OpenCV denoise h=5 → output PNG
```

No gamma correction is used in the final submission because gamma correction reduced validation PSNR.

---

## Final Model

The final selected checkpoint is:

```text
experiments/lowlight_nafnet_stage4_mse_lr5e6/models/net_g_500.pth
```

This checkpoint was also saved as:

```text
net_g_stage4_mse_best.pth
```

The model was fine-tuned from the previous best Stage 3 checkpoint at iteration 43000 using MSELoss.
The best validation result before post-processing was:

```text
Raw NAFNet PSNR: 21.0609 dB
Raw NAFNet SSIM: 0.6485
```

After OpenCV denoising h=5:

```text
NAFNet + denoise h=5 PSNR: 21.0774 dB
```

---

## Important Files Needed to Run

The important files/folders needed to run the final model are:

```text
basicsr/
options/train/lowlight_nafnet_stage4_mse_lr5e6.yml
run_final_model_h5.py
run_final_model_raw.py
count_final_model_size_flops.py
requirements.txt
setup.py
LICENSE
```

The final checkpoint is needed separately:

```text
experiments/lowlight_nafnet_stage4_mse_lr5e6/models/net_g_500.pth
```

or:

```text
net_g_stage4_mse_best.pth
```

---

## Experiments Folder

The `experiments/` folder contains saved training checkpoints from different training stages and ablations.

Important experiment folders:

```text
experiments/lowlight_nafnet_width32/
experiments/lowlight_nafnet_finetune_stage1/
experiments/lowlight_nafnet_finetune_stage2/
experiments/lowlight_nafnet_finetune_stage3_psnr/
experiments/lowlight_nafnet_stage4_mse_lr5e6/
```

These folders contain model checkpoints such as:

```text
net_g_34000.pth
net_g_43000.pth
net_g_500.pth
```

The final model checkpoint is from:

```text
experiments/lowlight_nafnet_stage4_mse_lr5e6/
```

Only the final checkpoint is required for inference. Older checkpoints are kept for ablation records and comparison.

---

## Dataset Format

The model expects input images in a folder. For training and validation, the dataset was prepared as:

```text
Datasets/nafnet_data/train/lq
Datasets/nafnet_data/train/gt
Datasets/nafnet_data/val/lq
Datasets/nafnet_data/val/gt
Datasets/nafnet_data/test/lq
```

For inference, only the input folder is needed, for example:

```text
Datasets/nafnet_data/test/lq
```

---

## How to Run Final Inference

Run the final model with OpenCV denoise h=5:

```bash
python run_final_model_h5.py \
  --config options/train/lowlight_nafnet_stage4_mse_lr5e6.yml \
  --weights experiments/lowlight_nafnet_stage4_mse_lr5e6/models/net_g_500.pth \
  --input_dir path/to/input_images \
  --output_dir outputs/denoise_h5 \
  --max_images 0
```

The output folder will contain final enhanced PNG images.

---

## Raw NAFNet Output

To run NAFNet without denoising:

```bash
python run_final_model_raw.py \
  --config options/train/lowlight_nafnet_stage4_mse_lr5e6.yml \
  --weights experiments/lowlight_nafnet_stage4_mse_lr5e6/models/net_g_500.pth \
  --input_dir path/to/input_images \
  --output_dir outputs/nafnet_raw \
  --max_images 0
```

---

## Model Size and FLOPs

Measured using a `256 × 256 × 3` input:

```text
Model size: 29.1597 M parameters
Conv/Linear MACs: 16.0450 GMACs
Elementwise FLOPs: 0.6459 GFLOPs
Total FLOPs: 32.7360 GFLOPs
Convention: 1 MAC = 2 FLOPs
```

OpenCV denoising has 0 trainable parameters. The reported model size is for the NAFNet model.

---

## Ablation Summary

| Experiment                   | Validation PSNR | Validation SSIM | Notes                            |
| ---------------------------- | --------------: | --------------: | -------------------------------- |
| Baseline NAFNet              |         18.8953 |          0.5803 | Best baseline checkpoint         |
| Stage 2 fine-tuning          |         20.7396 |          0.6405 | Improved with staged fine-tuning |
| Stage 3 PSNRLoss, iter 43000 |         20.8083 |          0.6456 | Previous best checkpoint         |
| Stage 4 MSELoss, iter 500    |         21.0658 |          0.6485 | Best raw NAFNet checkpoint       |
| Stage 4 MSE + denoise h=5    |         21.0774 |               — | Final selected pipeline          |

---

## Post-Processing Comparison

| Pipeline                          | Validation PSNR |
| --------------------------------- | --------------: |
| Raw NAFNet                        |      21.0609 dB |
| NAFNet + denoise h=5              |      21.0774 dB |
| NAFNet + denoise h=5 + gamma 0.80 |      19.5231 dB |

Denoising h=5 slightly improved PSNR. Gamma correction made images brighter visually, but reduced PSNR, so gamma correction was not used in the final pipeline.

---

## License

This project is based on NAFNet/BasicSR-style code from Megvii-Model and follows the included MIT License. We modified and fine-tuned the model for the low-light image enhancement task.
