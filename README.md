# HiC-SuperNet

HiC-SuperNet: Multi-Scale Attention-Based Deep Learning for High-Resolution Chromatin Contact Map Enhancement.

This repository runs HiC-SuperNet with the same workflow and data format as [DiCARN](https://github.com/OluwadareLab/DiCARN): `Read_Data.py` → `Downsample.py` → `Generate.py` → train → predict. The model is a PyTorch port of the original TensorFlow implementation ([OluwadareLab/HiC-SuperNet](https://github.com/OluwadareLab/HiC-SuperNet)), with the same layers, filter counts and loss.

## Architecture
- 7×7 feature-extraction convolution with BatchNorm and ReLU
- 8 Multi-Scale Dilated Residual Blocks (parallel 3×3 convolutions, dilation 1, 2, 4, with a 1×1 projected shortcut)
- Dual (channel + spatial) attention after every second block
- Global residual connection, then progressive refinement convolutions (128 → 64 → 32 filters) and a linear output layer
- 791,445 trainable parameters
- Loss: 0.4·MSE + 0.2·MAE + 0.3·(1 − Pearson) + 0.1·(1 − SSIM)

## Repository layout
```
Arg_Parser.py              root_dir and chromosome splits (set_dict)
Read_Data.py               raw Rao et al. matrices  -> Data/mat/<cell>/chrN_10kb.npz
Downsample.py              10kb -> 40kb (1/16 reads)  -> Data/mat/<cell>/chrN_40kb.npz
Generate.py                40x40 submatrices          -> Data/data/hicarn_*.npz
HiCSuperNet_Train.py       training
Predict_HiCSuperNet.py     prediction + SSIM / MSE / PSNR / GenomeDISCO
Models/HiCSuperNet_model.py
Utils/                     io, SSIM, GenomeDISCO, losses
checkpoints/HiCSuperNet/   trained weights are saved here
```

## Installation
Python 3.10 or newer.
```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
The device is chosen automatically: CUDA GPU, then Apple Silicon GPU (MPS), then CPU.

## Data
`root_dir` is set in `Arg_Parser.py` (default `./Data`).

**Quick start.** Put preprocessed DiCARN/HiCARN `.npz` files (for example from [Zenodo](https://zenodo.org/records/15198848)) directly in `Data/data/`. Files made for DiCARN work without changes.

**From raw data.** Download the intrachromosomal contact matrices from GEO [GSE63525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525) and extract them into `Data/raw/<cell_line>/` (for GM12878 this creates folders such as `10kb_resolution_intrachromosomal/chr1/MAPQGE30/`). Then:
```
python Read_Data.py -c GM12878
python Downsample.py -hr 10kb -lr 40kb -r 16 -c GM12878
python Generate.py -hr 10kb -lr 40kb -lrc 100 -s train        -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878
python Generate.py -hr 10kb -lr 40kb -lrc 100 -s valid        -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878
python Generate.py -hr 10kb -lr 40kb -lrc 100 -s GM12878_test -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878
```
The chromosomes used for each split are defined in `set_dict` in `Arg_Parser.py`. For other cell lines, run the same steps with `-c K562` (etc.) and `-s K562_test`.

## Training
Needs `hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz` and `..._valid.npz` in `Data/data/`.
```
python HiCSuperNet_Train.py
```
Options: `-e` epochs (100), `-b` batch size (16), `-lr` learning rate (1e-3, cosine decay), `-p` early-stopping patience (15), `--augment` (random flips), `--device` (auto/cuda/mps/cpu).

Outputs in `checkpoints/HiCSuperNet/`:
- `<date>_bestV_..._HiCSuperNet.pytorch` (lowest validation loss; use this for prediction)
- `<date>_finalV_..._HiCSuperNet.pytorch` (last epoch)

Per-epoch validation scores are written to `score_tracker/HiCSuperNet/`.

## Prediction with analysis
```
python Predict_HiCSuperNet.py -m HiCSuperNet -lr 40kb \
    -ckpt checkpoints/HiCSuperNet/<date>_bestV_10kb40kb_c40_s40_b201_nonpool_HiCSuperNet.pytorch \
    -f hicarn_10kb40kb_c40_s40_b201_nonpool_GM12878_test.npz \
    -c GM12878_HiCSuperNet
```
`-f` is a file name inside `Data/data/`. Per-chromosome and mean SSIM, MSE, PSNR and GenomeDISCO are printed. Enhanced maps are saved to `Data/predict/<-c>/predict_chrN_40kb.npz`, with the matrix under the `hicarn` key and non-zero indices under `compact`. If you trained with non-default `--base_filters` or `--num_blocks`, pass the same values here.

## Changes from the DiCARN code
- macOS / Windows: prediction no longer requires 24+ CPU cores and saves results without a multiprocessing pool (the pool failed silently on macOS).
- Removed NumPy aliases (`np.int`) that newer NumPy versions no longer support.
- `Read_Data.py` reads only the requested resolution and mapping quality, and preprocessing errors are printed instead of silently ignored.
- Prediction metrics are grouped per sample, so a batch that spans two chromosomes is scored correctly.
