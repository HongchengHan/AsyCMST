# AsyCMST: Asymmetric Cross-Modal Spatio-Temporal Learning for Multimodal Ultrasound Nodule Recognition

This repository contains the pre-released code for AsyCMST, a method for asymmetric cross-modal spatio-temporal learning applied to multimodal ultrasound nodule recognition. The model leverages both B-mode and Contrast-Enhanced Ultrasound (CEUS) data for improved diagnosis.

## Features

- Multimodal ultrasound analysis (B-mode and CEUS)
- Spatio-temporal learning approach
- Asymmetric cross-modal integration
- Pre-trained models and evaluation scripts

## Data

A subset of the multimodal ultrasound nodule diagnosis dataset can be downloaded from [XJTU-MMUS-subset-20260401.zip (Google Drive)](https://drive.google.com/file/d/1JQtKzFBBRXw9AyCcMGQ64Va-CnmCVmA5/view?usp=drive_link).

Place the downloaded data in the `datasets` directory. The videos are in the `videos` directory, and we extract selected frames from the videos to accelerate data loading during training. The extracted frames are in the `images` directory.

The expected directory structure is:

```
datasets/
└── XJTU-MMUS-subset-20260401/
    ├── images/
    │   ├── 0.ben/
    │   │   ├── bus/
    │   │   │   └── TT_119_CEUS_busvid/
    │   │   │       ├── 00000.jpg
    │   │   │       ├── 00001.jpg
    │   │   │       └── ...
    │   │   └── ceus/
    │   └── 1.mal/
    |── videos/
    |   ├── 0.ben/
    │   │   ├── bus/
    │   │   │   |── TT_119_CEUS_busvid.mp4
    │   │   │   |── TT_173_CEUS_busvid.mp4
    │   │   │   └── ...
    │   │   └── ceus/
    │   └── 1.mal/
    |── train_1.csv
    |── valid_1.csv
    └── test_1.csv
```

## Requirements

- Python 3.10+
- PyTorch
- Other dependencies (see `requirements.txt` if available, or install via pip)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HongchengHan/AsyCMST.git
   cd AsyCMST
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit the configuration files in the `config/` directory for your setup. Available configs:
- `train_asycmst_bus_ceus.yaml`
- `train_tsnet_rx50_bus_ceus.yaml`
- `test_asycmst_bus_ceus.yaml`

## Training

Run the training notebook:
- Open `train_model.ipynb` in Jupyter and execute the cells.

## Evaluation

Run the evaluation notebook:
- Open `eval_model.ipynb` in Jupyter and execute the cells.

## Models

Pre-trained models are available in the `models/` directory:
- `asycmst.py`: Main AsyCMST model
- `resnet.py`: ResNet backbone
- `tsnet.py`: TSNet model

## Citation

If you use this code, please cite:

```
Under review
```

## License

MIT license