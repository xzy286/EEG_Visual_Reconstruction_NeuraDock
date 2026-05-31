<h1 id="english">Neuradock-Vision-Reconstruction</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![EEG: Neuradock](https://img.shields.io/badge/Hardware-Neuradock-green.svg)](https://www.neuradock.com/)
<img width="2419" height="1516" alt="9e954b6acb2cbda3db0c0998750a0557" src="https://github.com/user-attachments/assets/0c90f7a2-b447-41d2-9557-e8430c6a177c" />

This repository contains the visual image reconstruction reproduction work based on the **NeuraDock** EEG device. The project covers the complete pipeline from raw signal acquisition and preprocessing to deep-learning-based image restoration.

> **Note**: The core reconstruction model is adapted from the paper: Li, Dongyang, et al. "Visual decoding and reconstruction via eeg embeddings with guided diffusion." arXiv preprint arXiv:2403.07721 (2024).

---

## Project Features

- **Data Acquisition**: Real-time signal recording and event marking tailored for the NeuraDock SDK.
- **Signal Processing**: Standardized EEG preprocessing pipeline based on MNE-Python (filtering, denoising, epoching).
- **Model Integration**: Feeds preprocessed EEG features into the visual reconstruction model from the paper.
- **End-to-End Workflow**: Complete closed loop from visual stimulus presentation to reconstructed image generation.

---

## Hardware & Environment Setup

### 1. Hardware Requirements
- **EEG Device**: Neuradock (7 channels)
- **Sampling Rate**: 250 Hz
- **Screen Refresh Rate**: 60 Hz

### 2. Software Environment

#### Processing & Training Environment (ML Environment)
Used for signal preprocessing, model training, and visual reconstruction.
- **Python Version**: 3.12 recommended
- **Installation**:
  ```bash
  conda create -n BCI python=3.12 -y
  conda activate BCI
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements_reconstruction.txt
  ```

---

## Repository Structure

```text
├── acquisition/                          # Data acquisition module
│   ├── config_rsvp.json                  # Place in the same directory as neuradock_rsvp.exe
│   └── neuradock_rsvp.exe                # Visual stimulation program (Psychopy) with signal reading & marking
├── preprocessing/                        # Signal preprocessing
│   └── neuradock_preprocessing.py        # Neuradock preprocessing; output aligned with the paper format
├── Generation/                           # Reconstruction model from the paper
├── Generation_adapters/
│   ├── eegdatasets_leaveone_neuradock.py # Replace Generation/eegdatasets_leaveone.py with this
│   ├── data_config.json                  # Replace Generation/data_config.json with this
│   └── ATMS_reconstruction_neuradock.py  # Replace Generation/ATMS_reconstruction.py with this
├── data/                                 # Data storage
└── notebooks/                            # Analysis & visualization demos
```

---

## Workflow

### Step 1: Environment Preparation & Data Acquisition

Before starting the experiment, ensure hardware connection and SDK configuration are complete:

1. **Install SDK**: Visit [Neuradock Downloads](https://neuradock.com/downloads) to download and install the latest **Neuradock SDK**.
2. **Device Connection**:
   - Wear the NeuraDock EEG device and connect it to the computer via USB.
   - Ensure good scalp-electrode contact (use the impedance check tool in the SDK).
3. **Start Data Port**:
   - Open the NeuraDock SDK.
   - After confirming the device is online, click **"Open Data Service"**, to start the TCP data forwarding service.
   <img width="3440" height="1369" alt="2624d61b6d4dd4644e36ff48b558917b" src="https://github.com/user-attachments/assets/7b9a02d7-f220-4be7-9337-379cf43684bb" />

   For device connection and data-port startup, refer to [NeuraDock Product Instructions](https://neuradock.com/product-instruction).

4. **Run Experiment Program**:
   Launch the visual stimulation script; it will automatically synchronize EEG acquisition and record `Marker` (event labels).
   neuradock_rsvp.exe download: https://pan.baidu.com/s/1jgqbQT_lgJ5KpZt6I5IBdg  extraction code: cvw4
   ```bash
   # Start RSVP (Rapid Serial Visual Presentation) experiment
   acquisition/neuradock_rsvp.exe
   ```

### Step 2: Preprocessing

Clean the acquired raw data and extract high-quality EEG features:

- **Filtering**: Apply 1–100 Hz band-pass and 50 Hz notch filters to remove power-line interference.
- **Artifact Removal**: Use the same mvnn algorithm as in the paper.
- **Epoching & Alignment**: According to the recorded `Marker` during acquisition, extract data segments after each visual stimulus and perform baseline correction.

```bash
# Convert raw data to the format consistent with the paper
# conda activate BCI
python preprocessing/neuradock_preprocessing.py
```

### Step 3: Image Reconstruction

Feed the preprocessed EEG signals into the deep-learning model to reproduce visual images:

1. **Adapter Replacement**:
   Replace `Generation/eegdatasets_leaveone.py` with `Generation_adapters/eegdatasets_leaveone_neuradock.py`;
   replace `Generation/data_config.json` with `Generation_adapters/data_config.json`;
   replace `Generation/ATMS_reconstruction.py` with `Generation_adapters/ATMS_reconstruction_neuradock.py`.
2. **Model Inference**: Run the training script

```bash
# conda activate BCI
python ATMS_reconstruction_neuradock.py --insubject True --subjects sub-001-preprocessed --logger True --gpu cuda:2 --output_dir ./outputs/contrast --train_file preprocessed_eeg_training__4__1__100.npy --test_file preprocessed_eeg_training__4__1__100_average.npy --test_name 100hz
```

---

## Results Preview

### 1. Quantitative Evaluation
We compared NeuraDock (7 channels) against the original paper (NeurIPS Paper, 128 channels full-brain/occipital). Results show that after 1.5x data augmentation, NeuraDock exhibits trends highly consistent with research-grade equipment on Top-k retrieval and N-way classification tasks.

<img width="1390" height="708" alt="0d881a4873e73eb674a1d572c1977fa2" src="https://github.com/user-attachments/assets/545c2f91-3301-4de6-956b-211cea6c2023" />

*Note: The figure shows NeuraDock under Equal Data and 1.5x Data modes, compared with the original paper's Occipital and Full-Brain classification accuracy.*

### 2. Qualitative Analysis: Reconstruction Showcase
Below are partial random-sample reconstruction results on the NeuraDock dataset.

| Ground Truth | NeuraDock Reconstructed |
| :---: | :---: |
| ![003df9db5df1de06fba33a403a65e8b6](https://github.com/user-attachments/assets/77468949-a430-4826-9b39-ff15e23941fa) | ![dd6eaaf05e12af24ba63cc2d46b26279](https://github.com/user-attachments/assets/f046896c-eb90-4fc2-adf5-299d0588cbcf) |

---

## Citation & Acknowledgements

### Core Model
The visual reconstruction algorithm in this project is adapted from:
Li, Dongyang, et al. "Visual decoding and reconstruction via eeg embeddings with guided diffusion." arXiv preprint arXiv:2403.07721 (2024).

[> [GitHub Link]](https://github.com/dongyangli-del/EEG_Image_decode/tree/main)

### Hardware Support
Thanks to the **Neuradock** team for device support and SDK documentation.

---

## License
This project is licensed under the [MIT License](LICENSE). Third-party model code copyrights belong to their respective authors.

---

### Contribution
If you discover preprocessing parameters better suited to the NeuraDock channel layout during reproduction, Pull Requests are welcome!

---


---

### 贡献建议
如果你在复现过程中发现了更适合 Neuradock 通道分布的预处理参数，欢迎提交 Pull Request！
