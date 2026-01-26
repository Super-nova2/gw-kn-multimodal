# Model_v1: GW-Optical Fusion (Pixel-Level GW Input)

## Overview
Model_v1 is a simplified GW-optical matching model. It removes the alignment/contrastive branch and keeps:
- A GW encoder (MLP) that consumes GW scalars plus matched skymap pixel features
- The optical encoder (mTAN + CLS token)
- A cross-attention fusion classifier for binary match/no-match prediction

The GW input no longer uses the full skymap sequence. Instead, for each optical transient, we match its sky coordinate to a skymap pixel, compute that pixel's credible level, and feed the per-pixel values as a compact GW feature vector.

## New GW Pixel Features
For each optical sample, we store `gw_pixel_features` with 8 values:
1. x
2. y
3. z
4. dA (pixel area)
5. dP (probability mass, scaled so sum = 100)
6. distmu
7. distsigma
8. credible_level (cumulative probability, 0..1)

The final GW input to the MLP is:
```
[gw_scalars (7)] + [gw_pixel_features (8)]
```

## Preprocessing
Generate a new HDF5 file with pixel-level GW features:
```
python /fred/oz016/bgao_kn/ML+GW+KN/Model_v1/preprocess_gw_pixel.py \
  --input_h5 /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset.h5 \
  --output_h5 /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_pixel.h5
```

For a quick local test (small subset):
```
python /fred/oz016/bgao_kn/ML+GW+KN/Model_v1/preprocess_gw_pixel.py \
  --input_h5 /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset.h5 \
  --output_h5 /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_pixel_small.h5 \
  --max_optical 2048
```

## Training (Local)
```
python /fred/oz016/bgao_kn/ML+GW+KN/Model_v1/train_v1.py \
  --data_path /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_pixel.h5 \
  --ckpt_path /fred/oz016/bgao_kn/data/model/checkpoints_v1 \
  --epochs 10 --batch_size 32 --steps_per_epoch 1000
```

### Mixed Sampling with Negative GW + Optical
Use mixed sampling to include negative GW events and extra optical negatives:
```
python /fred/oz016/bgao_kn/ML+GW+KN/Model_v1/train_v1.py \
  --data_path /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw_pixel.h5 \
  --neg_data_path /fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5 \
  --neg_group ELASTICC2_TRAIN/optical_data \
  --use_neg_gw --neg_gw_ratio 0.2 --samples_per_gw 4 \
  --ckpt_path /fred/oz016/bgao_kn/data/model/checkpoints_v1 \
  --epochs 10 --batch_size 32 --steps_per_epoch 1000
```

## Slurm Submission
Edit `Model_v1/args/train_v1.json` and submit:
```
/fred/oz016/bgao_kn/ML+GW+KN/Model_v1/train_v1.sh \
  /fred/oz016/bgao_kn/ML+GW+KN/Model_v1/args/train_v1.json
```

## Key Files
- `model_v1.py`: simplified fusion model (no alignment branch)
- `data_loader_v1.py`: dataset loader for GW pixel features
- `preprocess_gw_pixel.py`: HDF5 preprocessing to compute pixel features + credible levels
- `train_v1.py`: training loop for classification
- `train_v1.sh`: Slurm submit script
