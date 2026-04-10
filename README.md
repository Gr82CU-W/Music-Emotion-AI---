# Music Emotion AI - 情绪感知与音乐推荐系统

![Version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.12-green)

这是一个基于人工智能算法的情感计算与音乐推荐工具。系统能够通过多路文本情感模型识别用户心理状态，并在 Valence-Arousal (V-A) 空间内实时匹配最合适的音乐进行推荐。

## 📺 项目演示

您可以点击下载并观看本项目的功能演示视频：

**[查看演示视频 (demo.mp4)](demo.mp4)**

## 📂 项目结构

```text
.
├── run_app.py                # 程序主入口 (GUI 交互中心)
├── core/                     # 核心算法实现库
│   ├── engine_audio_va.py    # 自动化特征标注引擎 (基于 Playwright)
│   └── model_audio_cnn.py    # 音频特征回归网络 (CNN + SE-Block)
├── predict_va/               # 自然语言处理单元 (NLP Pipeline)
├── models/                   # 预训练权重存放目录
├── scripts/                  # 辅助工具集 (打包、评估、测试)
├── dataset_processed.csv     # 预置音乐情感特征数据集
├── audiolist/                # 分类管理的内置曲库
└── user_data/                # 用户数据与个性化私有数据库
```

## 🛠️ 技术路线与算法实现

本项目实现了一套从“情感语义感知”到“音乐特征匹配”的闭环技术体系。

### 1. 文本情感语义映射 (NLP Pipeline)
系统通过深度学习模型将非结构化的用户文字心情映射至二维情感空间坐标：
- **混合语言模型**：针对中文输入集成 `chinese-roberta-wwm-ext`，英文输入集成 `BERT-base-uncased`。
- **情感回归任务**：模型后端连接稠密层 (Dense Layers)，将高维语义特征压缩至二维向量，输出预测的愉悦度 ($V$) 与唤醒度 ($A$) 坐标。
- **实时推理逻辑**：利用 `predict_va` 单元实现快速情感捕获，并通过窗口平滑算法展示情感迁移过程。

### 2. 音频特征映射体系 (Audio Pipeline)
本项目采用**双引擎模式**，兼顾当前版本的准确性与未来演进：

- **方案 A：高精度自动化标注（当前稳定驱动）**
  - **实现逻辑**：利用 Playwright 自动化处理，将本地 MP3 上传至部署有音频数字信号处理 (DSP) 算法的 [Essentia.js 环境](http://mtg.github.io/essentia.js/examples/#/demos/)。
  - **数据闭环**：该引擎不仅用于实时标注用户导入的私有音乐，更承担了为“方案 B”生成大规模高质量 **Ground Truth** 训练集的关键任务。由于网页端算法具有极高的一致性，本版本优先采用此方案以确保推荐质量。

- **方案 B：端到端深度神经网络预测（预研演进模块）**
  - **核心模型**：内置基于卷积神经网络 (CNN) 的 `SimpleCNNEmotion` 模型结构。
  - **注意力机制**：引入 **Squeeze-and-Excitation (SE) Block** 通道注意力机制，能够自适应地对 Mel-Spectrogram (梅尔频谱图) 的不同频率通道进行重新加权。
  - **特征处理**：预处理阶段将音频提取为 128 维梅尔谱，并计算其一阶与二阶差分特征。待模型在方案 A 生成的数据集上完成充分训练后，即可替换方案 A 实现纯本地、无环境依赖的亚秒级离线感知。

### 3. 多策略音乐推荐算法 (Recommendation Engine)
推荐系统基于欧几里得空间几何逻辑实现，通过代码中的 `update_rec` 进行决策：
- **KNN 聚类检索**：在 V-A 空间内寻找离目标坐标最近的顶级候选音乐。
- **显著性加权因子**：引入计算公式 $(Dist - Magnitude \times 0.18)$。该算法不仅考虑距离，还会优先权重那些由于远离中性区而表现出更强烈情感特征（显著性高）的曲目，避免推荐结果趋于平淡。
- **三维场景适配逻辑**：
  - **Companion (陪伴模式)**：目标坐标 = 当前用户坐标。
  - **Regulate (调节模式)**：设定固定的舒适区坐标，强制引导负向心理状态向高愉悦、低唤醒转换。

## 🚀 快速开始

### 1. 环境准备 (推荐使用 Conda)
```bash
conda create -n music-ai python=3.12 -y
conda activate music-ai
```

### 2. 安装依赖与环境初始化
```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. 运行主程序
```bash
python run_app.py
```

## 📝 备注
本项目为“ISE3309 人工智能算法实践”课程作业成果。  
时间：2026年1月

