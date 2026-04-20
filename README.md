# MUMC
## [Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses for Medical Visual Question Answering](https://arxiv.org/abs/2307.05314)
This is the official implementation of `MUMC` for the medical visual question answering, which was accepted by [MICCAI-2023](https://conferences.miccai.org/2023/en/default.asp).
Our proposal achieves superior accuracy in comparison with other state-of-the-art (sota) methods on three public medical VQA datasets: [VQA-RAD dataset](https://www.nature.com/articles/sdata2018251#data-citations), [PathVQA dataset](https://arxiv.org/abs/2003.10286) and [Slake dataset](https://arxiv.org/abs/2102.09542). Paper link [here](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_36).

This repository is based on our previous [work](https://github.com/pengfeiliHEU/M2I2) and inspired by @Junnan Li's [work](https://github.com/salesforce/ALBEF). We sincerely thank for their sharing of the codes.

<div align=center>
<img src="fig/models.png" style="zoom:75%;">
</div>
<center>Figure 1: Overview of the proposed MUMC model. </center>

## Requirements
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Training and Testing
### 1. Dataset Preparation
Please organize the datasets as the following structure:
```angular2
+--clef2022
| +--train
| | +--ImageCLEFmedCaption_2022_train_000001.jpg
| | +--ImageCLEFmedCaption_2022_train_000002.jpg
| | +--...
| +--valid
| | +--ImageCLEFmedCaption_2022_valid_084258.jpg
| | +--ImageCLEFmedCaption_2022_valid_084259.jpg
| | +--...
| +--clef22022_train.json
| +--clef22022_valid.json

+--data_RAD
| +--images
| | +--synpic100132.jpg
| | +--synpic100176.jpg
| | +--...
| +--trainset.json
| +--testset.json
| +--answer_list.json

+--data_PathVQA
| +--images
| | +--train
| | | +--train_0000.jpg
| | | +--train_0001.jpg
| | | +--...
| | +--val
| | | +--val_0000.jpg
| | | +--val_0001.jpg
| | | +--...
| | +--test
| | | +--test_0000.jpg
| | | +--test_0001.jpg
| | | +--...
| +--pathvqa_test.json
| +--pathvqa_train.json
| +--pathvqa_val.json
| +--answer_trainval_list.json

+--data_Slake
| +--imgs
| | +--xmlab0
| | | +--source.jpg.jpg
| | | +--question.json
| | | +--...
| | +--....
| +--slake_test.json
| +--slake_train.json
| +--slake_val.json
| +--answer_list.json
```
### 2. Pre-training
```angular2
python3 pretrain  --output_dir ./pretrain
```

### 3. Finetune on Medical VQA tasks
```angular2
# choose medical vqa dataset(rad, pathvqa, slake)
python3 train_vqa.py --dataset_use rad --checkpoint ./pretrain/med_pretrain_29.pth  --output_dir ./output/rad
```

## Diffusion-Enhanced MUMC
This repository now keeps the original `MUMC` baseline path and adds an incremental diffusion-augmented variant inside the same training/evaluation framework.

### Added modules
- `DiffusionFeatureEncoder`: extracts 3 default multi-scale diffusion features and projects them to `256`.
- `ScaleGating`: a lightweight question-guided MLP that scores diffusion scales without self-attention.
- `DiffusionAligner`: a question-token to diffusion-token cross-attention block (`dim=256`, `heads=8`, `layers=1` by default).
- `GatedVisualFusion`: fuses diffusion-aware representations back into the original MUMC visual token stream before the existing MUMC multimodal encoder.
- Auxiliary losses:
  - global image-question InfoNCE contrastive loss
  - optional answer-semantic contrastive loss
  - multi-scale decorrelation loss

### Design notes
- MUMC remains the backbone VQA framework.
- The original visual encoder, text encoder, text decoder, training loop, evaluation pipeline, and dataset loaders are reused.
- Diffusion is an auxiliary representation/alignment enhancer, not a generator during VQA inference.
- The diffusion backbone is frozen by default because medical VQA datasets are small.
- When `diffusers` weights are unavailable, the code falls back to a frozen convolutional pyramid stub so ablation code paths remain runnable.

### Variants / ablations
- `mumc`: original MUMC baseline.
- `mumc_diffrep`: MUMC + diffusion representation fusion only.
- `mumc_diffalign`: MUMC + question-guided diffusion aligner only.
- `mumc_diffrepalign`: full model with both branches.

The variant is controlled by `--model_variant` or `configs/VQA.yaml`.

### Recommended training stages
- `baseline`: original MUMC training.
- `diffusion`: freeze most MUMC visual/text backbone parameters and train the new diffusion modules + decoder head.
- `joint`: unfreeze the last few visual/text layers for light joint fine-tuning.

### Example commands
```bash
# 1) Original MUMC baseline
python3 train_vqa.py \
  --dataset_use rad \
  --checkpoint ./pretrain/med_pretrain_29.pth \
  --output_dir ./output/rad_baseline \
  --model_variant mumc

# 2) MUMC + DiffRep
python3 train_vqa.py \
  --dataset_use rad \
  --checkpoint ./pretrain/med_pretrain_29.pth \
  --output_dir ./output/rad_diffrep \
  --model_variant mumc_diffrep \
  --train_stage diffusion

# 3) MUMC + DiffAlign
python3 train_vqa.py \
  --dataset_use rad \
  --checkpoint ./pretrain/med_pretrain_29.pth \
  --output_dir ./output/rad_diffalign \
  --model_variant mumc_diffalign \
  --train_stage diffusion

# 4) Full MUMC + DiffRep + DiffAlign
python3 train_vqa.py \
  --dataset_use rad \
  --checkpoint ./pretrain/med_pretrain_29.pth \
  --output_dir ./output/rad_diffrepalign \
  --model_variant mumc_diffrepalign \
  --train_stage diffusion \
  --use_global_contrast true \
  --use_decorrelation true \
  --new_lr 1e-4 \
  --backbone_lr 1e-5
```

### Evaluation
```bash
python3 train_vqa.py \
  --dataset_use rad \
  --checkpoint ./output/rad_diffrepalign/<checkpoint>.pth \
  --output_dir ./output/rad_diffrepalign_eval \
  --model_variant mumc_diffrepalign \
  --evaluate
```

### Useful ablations
- Disable global contrastive loss: `--use_global_contrast false`
- Disable decorrelation loss: `--use_decorrelation false`
- Enable answer-semantic contrastive loss: `--use_answer_contrast true`
- Single-scale diffusion: set `diffusion.active_scales` to a single index in `configs/VQA.yaml`
- Light joint fine-tuning: `--train_stage joint --unfreeze_visual_last_n_blocks 2 --unfreeze_text_last_n_layers 2`

### Optional diffusion dependency
If you want to use a real diffusion backbone instead of the built-in frozen stub, install `diffusers` and make sure the model weights referenced by `diffusion.model_id` are available locally or downloadable in your environment.

## Comparison with the sota
<img src="fig/results.png">

## Pretrained weights
You can download the pre-trained weights through the following [link](https://drive.google.com/file/d/1ZxwjfDeBYTMpw4mN_R9-gAcR9UOJiAFb/view?usp=sharing).

## Citation:
```
@article{MUMC,
  title     = {Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses for Medical Visual Question Answering},
  author    = {Pengfei Li, Gang Liu, Jinlong He, Zixu Zhao and Shenjun Zhong},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year      = {2023},
  pages     = {374--383},
  publisher = {Springer Nature Switzerland}
}
```

## License
MIT License
