# GLDAS-Gaze

GLDAS-Gaze: Global-Local Dynamic Adaptive Synergy for Task-Driven Scanpath Prediction
Authors: Shangyu Zhou, Dongbo Zhang, Jianwu Fang, and Yaonan Wang.

---

## Abstract

This repository provides the code for training and evaluating the GLDAS-Gaze model for task-driven scanpath prediction. The model uses a dual-stream (dorsal/ventral) architecture with cross-stream interaction and supports both Target-Present (TP) and Target-Absent (TA) evaluation on COCO-Search18.

---

## Environment Setup

### 1. Create conda environment

We use the conda environment named `gldas-gaze` (Python 3.9). Create and activate it:

```bash
conda create -n gldas-gaze python=3.9 -y
conda activate gldas-gaze
```

### 2. Install PyTorch (CUDA 11.x)

```bash
# Example for CUDA 11.7; adjust according to your driver.
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3. Install detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Or from local clone:
# git clone https://github.com/facebookresearch/detectron2.git && pip install -e detectron2
```

### 4. Install remaining dependencies

```bash
cd /path/to/GLDAS-Gaze
pip install -r requirements.txt
```

### 5. (Optional) Multi-Scale Deformable Attention CUDA op

For faster training, you can compile the CUDA extension (otherwise a PyTorch fallback is used):

```bash
cd gldas_gaze/pixel_decoder/ops
sh make.sh
cd ../../..
```

---

## Data Preparation

### COCO-Search18

1. Obtain [COCO-Search18](https://github.com/cvlab-stonybrook/COCO-Search18) and place the dataset under a root directory, e.g. `dataset/`.
2. Expected structure under `dataset_root`:

   - `coco_search_fixations_512x320_on_target_allvalid.json` — scanpath annotations
   - `bbox_annos.npy` — bounding box annotations
   - `semantic_seq_full/` — semantic / segmentation data (if used)
   - Other files as required by `common/dataset.py` and `gldas_gaze/builder.py`

3. Set `--dataset-root` to this path in all commands below.

### Pretrained backbone and pixel decoder

- Place ResNet-50 backbone weights and MSDeformAttn pixel decoder weights in a directory (e.g. `pretrained_models/`).
- **Backbone**: e.g. `M2F_R50.pkl` (Mask2Former-style ResNet-50).
- **Pixel decoder**: e.g. `M2F_R50_MSDeformAttnPixelDecoder.pkl`.

Edit `configs/resnet50.yaml` and set `WEIGHTS` to the path of your backbone file, e.g.:

```yaml
MODEL:
  WEIGHTS: "pretrained_models/M2F_R50.pkl"
```

The builder loads the pixel decoder from the same directory (see `gldas_gaze/builder.py`). If your paths differ, adjust the paths in the config or builder accordingly.

---

## Training

Activate the environment and run from the project root:

```bash
conda activate gldas-gaze
cd /path/to/GLDAS-Gaze
```

### Target-Present (TP)

```bash
python train.py \
  --hparams configs/coco_search18_dense_SSL_TP.json \
  --dataset-root /path/to/your/dataset \
  --gpu-id 0
```

### Target-Absent (TA)

```bash
python train.py \
  --hparams configs/coco_search18_dense_SSL_TA.json \
  --dataset-root /path/to/your/dataset \
  --gpu-id 0
```

- Checkpoints and logs are written to the path in `Train.log_dir` in the corresponding JSON config (e.g. `./assets/GLDAS_TP` or `./assets/GLDAS_TA`).
- Training iterations and other hyperparameters are set in the same JSON files.

---

## Evaluation

To evaluate a trained checkpoint (e.g. TP or TA):

```bash
conda activate gldas-gaze
python test_eval.py \
  --hparams configs/coco_search18_dense_SSL_TP.json \
  --dataset-root /path/to/your/dataset \
  --ckpt /path/to/checkpoint.pt \
  --gpu-id 0 \
  --eval-mode greedy \
  --tap TP
```

- Use `--tap TP` or `--tap TA` to evaluate on Target-Present or Target-Absent, respectively.
- `--eval-mode greedy` uses greedy decoding; use `sample` for sampling-based decoding.

---

## Configuration

- TP: `configs/coco_search18_dense_SSL_TP.json` 
- TA: `configs/coco_search18_dense_SSL_TA.json` 

Backbone and data paths are set in `configs/resnet50.yaml` and the JSON configs.

---

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{zhou2025gldasgaze,
  title        = {GLDAS-Gaze: Global-Local Dynamic Adaptive Synergy for Task-Driven Scanpath Prediction},
  author       = {Zhou, Shangyu and Zhang, Dongbo and Fang, Jianwu and Wang, Yaonan},
  year         = {2025},
  journal      = {},
  url          = {https://github.com/ZSY-humble/GLDAS-Gaze}
}
```

---

## Reproducibility (one-shot)

With conda and the dataset/weights in place, you can run:

```bash
conda create -n gldas-gaze python=3.9 -y && conda activate gldas-gaze
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
# Optional: cd gldas_gaze/pixel_decoder/ops && sh make.sh && cd ../../..
# Set dataset root and config paths, then:
python train.py --hparams configs/coco_search18_dense_SSL_TP.json --dataset-root /path/to/dataset --gpu-id 0
python test_eval.py --hparams configs/coco_search18_dense_SSL_TP.json --dataset-root /path/to/dataset --ckpt ./assets/GLDAS_TP/ckp_XXXX.pt --gpu-id 0 --eval-mode greedy --tap TP
```

---

## License

This project is for research use. Please refer to the licenses of COCO-Search18, detectron2, and Deformable DETR when using the code and data.
