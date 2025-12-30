# Neuradock-Vision-Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![EEG: Neuradock](https://img.shields.io/badge/Hardware-Neuradock-green.svg)](https://www.neuradock.com/)
<img width="2419" height="1516" alt="9e954b6acb2cbda3db0c0998750a0557" src="https://github.com/user-attachments/assets/0c90f7a2-b447-41d2-9557-e8430c6a177c" />





本仓库包含基于 **Neuradock** 脑电设备的视觉图像重建复现工作。项目涵盖了从原始信号采集、预处理到接入深度学习模型进行图像还原的完整流程。

> **说明**：本项目核心重建模型参考自论文：Li, Dongyang, et al. "Visual decoding and reconstruction via eeg embeddings with guided diffusion." arXiv preprint arXiv:2403.07721 (2024).

---

## 📌 项目特性

- **数据采集**：适配 Neuradock SDK 的实时信号记录与打标（Event Marking）。
- **信号处理**：基于 MNE-Python 的标准化脑电预处理流水线（滤波、去噪、分段）。
- **模型集成**：将预处理后的 EEG 特征对接到论文的视觉重建模型。
- **端到端流程**：从视觉刺激呈现到重建图像生成的完整闭环。

---

## 🛠 硬件与环境准备

### 1. 硬件需求
- **脑电设备**：Neuradock ( 7通道 )
- **采样率**：250Hz
- **屏幕刷新率**：60Hz

### 2. 软件环境

#### 处理与训练环境 (ML Environment)
用于信号预处理、模型训练及视觉重建。
- **Python 版本**: 建议 3.12 
- **安装指令**:
  ```bash
  conda create -n BCI python=3.12 -y
  conda activate BCI
  pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements_reconstruction.txt
  ```
---

## 📂 仓库目录结构

```text
├── acquisition/                          # 数据采集模块
│   ├── config_rsvp.json                  # 与neuradock_rsvp.exe放在同一个目录下
│   └── neuradock_rsvp.exe                # neuradock设备的视觉刺激程序 (Psychopy)，包含信号读取与打标
├── preprocessing/                        # 信号预处理
│   └── neuradock_preprocessing.py        # neuradock设备预处理，处理后的数据格式与论文对齐
├── Generation/                           # 论文的重建模型
├── Generation_adapters/          
│   ├── eegdatasets_leaveone_neuradock.py # 用这个文件替换掉Generation中的eegdatasets_leaveone.py文件
│   ├── data_config.json                  # 用这个文件替换掉Generation中的data_config.json文件
│   └── ATMS_reconstruction_neuradock.py  # 用这个文件替换掉Generation中的ATMS_reconstruction.py文件
├── data/                                 # 数据存放
└── notebooks/                            # 过程分析与可视化 Demo
```

---


### 🚀 工作流程 (Workflow)

#### 第一步：环境准备与数据采集 (Acquisition)

在开始实验前，请确保已完成硬件连接与 SDK 环境配置：

1.  **安装 SDK**：访问 [Neuradock 官网https://neuradock.com/downloads] 下载并安装最新的 **Neuradock SDK**。
2.  **设备连接**：
    *   佩戴 Neuradock 脑电设备，并确保电极与头皮接触良好（建议通过 SDK 自带工具检查阻抗）。
    *   通过蓝牙或专用接收器将设备连接至电脑。
3.  **启动数据端口**：
    *   打开 Neuradock SDK 。
    *   确认设备在线后，点击 **"打开数据服务"**，开启数据转发服务（TCP协议）。
    <img width="3440" height="1369" alt="60f8725bb3625d3a1824ff32fc06a994" src="https://github.com/user-attachments/assets/1b044fca-8834-40e4-b8de-cda2d323a3c2" />

设备连接和启动数据端口可以参考 [https://neuradock.com/product-instruction]

4.  **运行实验程序**：
    运行视觉刺激脚本，程序将自动同步采集脑电信号并记录 `Marker`（事件标记）。
    neuradock_rsvp.exe 下载链接: https://pan.baidu.com/s/1jgqbQT_lgJ5KpZt6I5IBdg 提取码: cvw4
    ```bash
    # 启动 RSVP (快速序列视觉呈现) 实验
    acquisition/neuradock_rsvp.exe
    ```

#### 第二步：预处理 (Preprocessing)

对采集到的原始信号（Raw Data）进行清洗，提取高质量的脑电特征：

- **滤波处理**：应用 1 - 100 Hz 带通滤波及 50 Hz 陷波滤波以去除工频干扰。
- **伪迹去除**：采用与论文相同的mvnn算法。
- **分段与对齐**：根据采集时记录的 `Marker`，截取每个视觉刺激后 [X]ms 的数据段，并进行基线校正。

```bash
# 将原始数据转换为与论文中一致的格式
# conda activate BCI
python preprocessing/neuradock_preprocessing.py
```

#### 第三步：图像重建 (Reconstruction)

将预处理后的 EEG 信号输入深度学习模型，复现视觉图像：

1.  **适配转换**：
    用Generation_adapters/eegdatasets_leaveone_neuradock.py替换掉Generation中的eegdatasets_leaveone.py文件；用Generation_adapters/data_config.json替换掉Generation中的data_config.json文件；用Generation_adapters/ATMS_reconstruction_neuradock.py替换掉Generation中的ATMS_reconstruction.py文件。
4.  **模型推理**：运行训练脚本

```bash
# conda activate BCI
python ATMS_reconstruction_neuradock.py --insubject True --subjects sub-001-preprocessed --logger True --gpu cuda:2  --output_dir ./outputs/contrast --train_file preprocessed_eeg_training__4__1__100.npy --test_file preprocessed_eeg_training__4__1__100_average.npy --test_name 100hz 
```

---

## 📊 实验结果预览 (Results)

### 1. 定量对比 (Quantitative Evaluation)
我们将 Neuradock (7通道) 的表现与原论文 (NeurIPS Paper, 128通道全脑/枕叶) 进行了对比。结果显示，在经过 1.5 倍数据量增强后，Neuradock 在 Top-k 检索和 N-way 分类任务上表现出与科研级设备高度一致的趋势。

<img width="1390" height="708" alt="0d881a4873e73eb674a1d572c1977fa2" src="https://github.com/user-attachments/assets/ba15a650-3f82-4ec5-8287-8cc5d17e67e8" />


*注：图中展示了 Neuradock 在 Equal Data (等量数据) 和 1.5x Data (增强数据) 模式下，对比原论文 Occipital (枕叶) 与 Full-Brain (全脑) 的分类准确率。*

### 2. 定性分析：图像重建展示 (Qualitative Reconstruction)
以下展示了模型在 Neuradock 数据集上的部分随机采样重建结果。

| 原始视觉刺激 (Ground Truth) | Neuradock 重建图像 (Reconstructed) |
| :---: | :---: |
| ![003df9db5df1de06fba33a403a65e8b6](https://github.com/user-attachments/assets/77468949-a430-4826-9b39-ff15e23941fa) | ![dd6eaaf05e12af24ba63cc2d46b26279](https://github.com/user-attachments/assets/f046896c-eb90-4fc2-adf5-299d0588cbcf) |
---

## 📜 引用与致谢

### 核心模型
本项目视觉重建算法部分代码引用/修改自以下研究：
Li, Dongyang, et al. "Visual decoding and reconstruction via eeg embeddings with guided diffusion." arXiv preprint arXiv:2403.07721 (2024).

[> [GitHub 链接]](https://github.com/dongyangli-del/EEG_Image_decode/tree/main)

### 硬件支持
感谢 **Neuradock** 团队提供的设备支持与 SDK 接口文档。

---

## ⚖️ 许可证
本项目采用 [MIT License](LICENSE) 授权。第三方模型代码版权归原作者所有。

---

### 💡 贡献建议
如果你在复现过程中发现了更适合 Neuradock 通道分布的预处理参数，欢迎提交 Pull Request！

---
