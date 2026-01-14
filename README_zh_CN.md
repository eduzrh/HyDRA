<div align="center">

<h1>

✨ Towards Temporal Knowledge Graph Alignment in the Wild ✨

</h1>



<h3>—————— 投稿于 IEEE TPAMI ——————</h3>

</div>



<div align="center">

[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/eduzrh/HyDRA)
[![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/eduzrh/HyDRA/issues)

[English](README.md) | [简体中文](./README_zh_CN.md)

</div>



<p align="center">
  <a href="#-introduction"><b>📰 简介</b></a> |
  <a href="#architecture"><b>🏗️ 架构</b></a> |
  <a href="#installation"><b>⚙️ 安装</b></a> |
  <a href="#-quick-start"><b>🚀 快速开始</b></a> <br>
  <a href="#-datasets"><b>📦 数据集</b></a> |
  <a href="#-usage"><b>📖 使用说明</b></a> |
  <a href="#-reproducibility"><b>🔬 可复现性</b></a> |
  <a href="#-license"><b>📜 许可证</b></a> |
  <a href="#-contact"><b>📬 联系方式</b></a>
</p>



---

## 📰 最新动态

<div align="center">

| 🆕 更新 | 📅 日期 | 📝 描述 |
|:---:|:---:|:---|
| 🎉 **代码发布** | - | HyDRA 代码库和数据集现已可用 |

</div>

---

# 📰 简介



**真实场景下的时序知识图谱对齐（TKGA-Wild）** 解决了时序知识图谱集成中的一个关键挑战。据我们所知，这是**首个**正式提出并解决该问题的工作，我们将其称为 **TKGA-Wild**。由于**多尺度时序元素**（即多粒度时序共存和时序跨度差异）和**非对称时序结构**（即异构时序结构和时序结构不完整性）在真实场景中普遍存在，该任务面临着独特的挑战。



为了弥合这一差距，我们提出了 **HyDRA**，这是一种基于**多尺度超图检索增强生成**的新范式，以系统性地解决 TKGA-Wild 的独特挑战。HyDRA 有效捕获复杂的结构依赖关系，建模多粒度时序特征，缓解时序差异，并引入了一种新的**尺度交织协同机制**来协调不同时序尺度的信息。



## 🔥 核心特性



<div align="center">

| 特性 | 图标 | 描述 |
|:---|:---:|:---|
| **多粒度时序编码** | 🔄 | 在不同尺度（年、月、日）捕获时序信息 |
| **尺度自适应实体投影** | 📐 | 跨不同图尺度和维度的自适应实体投影 |
| **多尺度超图检索** | 🔍 | 基于超图的高效神经检索 |
| **尺度交织协同** | 🔗 | 协调不同时序尺度的信息 |
| **最先进的性能** | 📈 | 持续超越 28 个竞争基线，在 Hits@1 上实现高达 43.3% 的提升 |

</div>



---

## 🏗️ 架构



HyDRA 采用**多尺度超图检索增强生成**范式，包含以下几个关键阶段：



阶段 1：编码与集成 🔄



阶段 2：尺度自适应实体投影 📐

阶段 3：多尺度超图检索 🔍


阶段 4：多尺度融合 🔗



> 📖 有关详细的架构描述和理论基础，请参考随附的论文。



---

## ⚙️ 安装



### 📋 前置要求



首先，安装依赖项：



```bash

pip install -r requirements.txt

```



### 📦 主要依赖



| 包 | 版本 | 用途 |
|:---|:---:|:---|
| 🐍 **Python** | >= 3.7 | 核心语言（测试于 3.8.10） |
| 🔥 **PyTorch** | >= 1.10.0 | 深度学习框架 |
| 🔍 **Faiss** | >= 1.7.0 | 高效相似性搜索（CPU/GPU） |
| 📊 **NumPy** | >= 1.21.0 | 数值计算 |
| 🐼 **Pandas** | >= 1.3.0 | 数据处理 |
| ⏳ **Tqdm** | >= 4.62.0 | 进度条 |
| 🌐 **NetworkX** | >= 2.6.0 | 图分析 |



> 💡 **注意：** 对于 GPU 加速的 FAISS，请使用 `faiss-gpu` 而不是 `faiss-cpu`。



---

## 📦 数据集



对于我们新提出的 **TKGA-Wild** 场景，我们引入了两个新的基准数据集：**BETA** 和 **WildBETA**。

<div align="center">

| 数据集 | 描述 | 事实规模 |
|:---|:---|:---|
| **BETA** | TKGA-Wild 的基准数据集 | 362K+ |
| **WildBETA** | TKGA-Wild 的扩展基准数据集 | 563K+ |

</div>

### 🔗 下载链接

<div align="center">

[![Baidu Netdisk](https://img.shields.io/badge/Baidu_Netdisk-Download-blue?style=for-the-badge)](https://pan.baidu.com/s/1TKZvjsDgqUrOAGKe6MRf9A?pwd=pnax)
[![Google Drive](https://img.shields.io/badge/Google_Drive-Download-green?style=for-the-badge)](https://drive.google.com/drive/folders/1P-YtGgoEh_y2RwKTS-YeM0X1sdWlWDEV?usp=sharing)

</div>

> 🔐 **百度网盘**：提取码：`pnax` | 密码：`tkgawild`

**数据集格式：**



以数据集 `icews_wiki` 为例，文件夹 `data/icews_wiki/` 应包含：



- `ent_ids_1`: 源知识图谱中的实体 ID

- `ent_ids_2`: 目标知识图谱中的实体 ID

- `triples_1`: 源知识图谱中由 ID 编码的关系三元组

- `triples_2`: 目标知识图谱中由 ID 编码的关系三元组

- `rel_ids_1`: 源知识图谱中的关系 ID

- `rel_ids_2`: 目标知识图谱中的关系 ID

- `time_id`: 源知识图谱和目标知识图谱中的时间 ID

- `ref_ent_ids`: 所有对齐的实体对，格式为 `(e_s \t e_t)` 的配对列表



**注意：** 实验中使用的代表性数据集来源于 [Dual-AMN](https://github.com/MaoXinn/Dual-AMN)、[JAPE](https://github.com/nju-websoft/JAPE)、[GCN-Align](https://github.com/1049451037/GCN-Align)、[BETA](https://github.com/DexterZeng/BETA)、[DAEA](https://github.com/yangxiaoxiaoly/DAEA)、[AGROLD, DOREMUS](https://github.com/EnsiyehRaoufi/Create_Input_Data_to_EA_Models) 及相关工作。



---

## 🚀 快速开始



### 步骤 1：克隆仓库 📥



```bash

git clone https://github.com/eduzrh/HyDRA.git

cd HyDRA

```



### 步骤 2：准备数据集 📦



下载并解压数据集到 `./data/`



### 步骤 3：运行主实验 ▶️



```bash

python HyDRA_main.py --data_dir data/icews_wiki

```



### 步骤 4：查看结果 📊



| 指标 | 描述 |
|:---|:---|
| **Hits@1** | 排名第一的正确对齐比例 |
| **Hits@10** | 前 10 名候选中的比例 |
| **MRR** | 平均倒数排名 |

---

## 📖 使用说明



### 基本用法



**运行完整流程：**



```bash

python HyDRA_main.py --data_dir data/icews_wiki

```



### 高级选项



**配置训练参数：**



```bash

python HyDRA_main.py --data_dir data/icews_wiki \
    --cuda 0 \
    --epochs 1500 \
    --max_iterations 5 \
    --min_kg1_entities 100

```



**参数说明：**



| 参数 | 类型 | 默认值 | 描述 |
|:---|:---:|:---:|:---|
| `--data_dir` | str | **必需** | 数据集目录路径 |
| `--skip_s4` | flag | False | 跳过编码阶段（如果结果已存在） |
| `--only_s4` | flag | False | 仅运行编码阶段 |
| `--cuda` | int | 0 | 用于训练的 CUDA 设备 ID |
| `--epochs` | int | 500 | 编码阶段的训练轮数 |
| `--max_iterations` | int | 3 | 最大流程迭代次数 |
| `--min_kg1_entities` | int | 50 | 停止的最小实体阈值 |



### 多粒度时间建模



HyDRA 支持多粒度时序建模（年和月级别）以处理多粒度时序共存。此功能可通过编码阶段配置启用。



---

## 🔬 可复现性



我们致力于确保结果的完全可复现性。提供以下资源：



### 📋 实验配置



- **超参数**：所有超参数设置都在代码中记录，可通过命令行参数配置

- **随机种子**：种子配置嵌入在训练脚本中以确保可复现性

- **环境**：在 Python 3.8.10 上测试，依赖项如 `requirements.txt` 中指定



### 📊 复现主要结果



要复现论文中报告的主要实验结果：



1. **下载数据集**，按照数据集部分描述的格式

2. **使用默认设置运行完整流程**：



```bash

python HyDRA_main.py --data_dir data/icews_wiki

```



3. **评估结果**，使用 `data/icews_wiki/message_pool/` 中的输出文件



### 🏗️ 代码组织



代码库组织为模块化组件以便清晰：



- `encoding_and_integration/`: 多粒度时序实体编码和集成

- `scale_adaptive_entity_projection/`: 关系对齐和实体投影

- `multi_scale_hypergraph_retrieval/`: 神经检索和超图分解

- `multi_scale_fusion/`: 多尺度融合和对齐细化

- `HyDRA_main.py`: 主流程编排器



### 📝 文档



- 全面的内联代码注释，解释关键设计决策

- 清晰的模块结构，采用标准化命名约定

- 本 README，包含逐步使用说明



---

## 📊 评估指标



我们采用标准的知识图谱对齐指标以确保透明度和可比性：



- **Hits@1**：排名第一的正确对齐比例

- **Hits@10**：前 10 名候选中的正确对齐比例  

- **MRR（平均倒数排名）**：正确对齐的平均倒数排名



## 📬 联系方式



- **邮箱**：[runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)

- **GitHub Issues**：对于技术问题，请在 [GitHub 仓库](https://github.com/eduzrh/HyDRA/issues) 中创建 Issue。标签：`bug`、`enhancement`、`question`。



目标在 2-3 个工作日内回复。



## 📜 许可证



[MIT License](LICENSE) - 保留版权声明。



---

## 🔗 参考文献

* [Unsupervised Entity Alignment for Temporal Knowledge Graphs](https://doi.org/10.1145/3543507.3583381).
  Xiaoze Liu, Junyang Wu, Tianyi Li, Lu Chen, and Yunjun Gao.
  Proceedings of the ACM Web Conference (WWW), 2023.
* [BERT-INT: A BERT-based Interaction Model for Knowledge Graph Alignment](https://doi.org/10.1145/3543507.3583381).
  Xiaobin Tang, Jing Zhang, Bo Chen, Yang Yang, Hong Chen, and Cuiping Li.
  Journal of Artificial Intelligence Research, 2020.
* [Benchmarking Challenges for Temporal Knowledge Graph Alignment](https://api.semanticscholar.org/CorpusID:273501043).
  Weixin Zeng, Jie Zhou, and Xiang Zhao.
  Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), 2024.
* [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks](https://doi.org/10.18653/v1/d18-1032).
  Zhichun Wang, Qingsong Lv, Xiaohan Lan, and Yu Zhang.
  Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.
* [Boosting the Speed of Entity Alignment 10×: Dual Attention Matching Network with Normalized Hard Sample Mining](https://doi.org/10.1145/3442381.3449897).
  Xin Mao, Wenting Wang, Yuanbin Wu, and Man Lan.
  Proceedings of the Web Conference (WWW), 2021.
* [Wikidata: A Free Collaborative Knowledgebase](https://doi.org/10.1145/2629489).
  Denny Vrandecic and Markus Krötzsch.
  Communications of the ACM, 2014.
* [Toward Practical Entity Alignment Method Design: Insights from New Highly Heterogeneous Knowledge Graph Datasets](https://doi.org/10.1145/3589334.3645720).
  Xuhui Jiang, Chengjin Xu, Yinghan Shen, Yuanzhuo Wang, Fenglong Su, Zhichao Shi, Fei Sun, Zixuan Li, Jian Guo, and Huawei Shen.
  Proceedings of the ACM Web Conference (WWW), 2024.
* [Unlocking the Power of Large Language Models for Entity Alignment](https://aclanthology.org/2024.acl-long.408).
  Xuhui Jiang, Yinghan Shen, Zhichao Shi, Chengjin Xu, Wei Li, Zixuan Li, Jian Guo, Huawei Shen, and Yuanzhuo Wang.
  Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2024.
* [Bootstrapping Entity Alignment with Knowledge Graph Embedding](https://doi.org/10.24963/ijcai.2018/611).
  Zequn Sun, Wei Hu, Qingheng Zhang, and Yuzhong Qu.
  Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2018.
* [NetworkX: Network Analysis in Python](https://github.com/networkx/networkx).
  NetworkX Developers.
  GitHub Repository.
* [Faiss: A Library for Efficient Similarity Search and Clustering of Dense Vectors](https://github.com/facebookresearch/faiss).
  Facebook Research.
  GitHub Repository.
* [DAEA: Enhancing Entity Alignment in Real-World Knowledge Graphs Through Multi-Source Domain Adaptation](https://aclanthology.org/2025.coling-main.393/)
  Linyan Yang, Shiqiao Zhou, Jingwei Cheng, Fu Zhang, Jizheng Wan, Shuo Wang, Mark Lee.
  COLING 2025
* [TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs](https://arxiv.org/abs/2406.09639)
  Julia Gastinger, Shenyang Huang, Mikhail Galkin, Erfan Loghmani, Ali Parviz, Farimah Poursafaei, Jacob Danovitch, Emanuele Rossi, Ioannis Koutis, Heiner Stuckenschmidt, Reihaneh Rabbany, Guillaume Rabusseau.
  NeurIPS 2024 Track on Datasets and Benchmarks



## 🙏 致谢

以下开源项目在本工作中被部分引用。我们真诚地感谢他们的贡献：

[Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [Simple-HHEA](https://github.com/jxh4945777/Simple-HHEA), [BETA](https://github.com/DexterZeng/BETA), [Dual-Match](https://github.com/ZJU-DAILY/DualMatch/), [Faiss](https://github.com/facebookresearch/faiss), [NetworkX](https://github.com/networkx/networkx), [AdaCoAgentEA](https://github.com/eduzrh/AdaCoAgentEA), [DAEA](https://github.com/yangxiaoxiaoly/DAEA), [AGROLD, DOREMUS](https://github.com/EnsiyehRaoufi/Create_Input_Data_to_EA_Models)

---

本仓库对应论文 ***Towards Temporal Knowledge Graph Alignment in the Wild***（投稿于 *IEEE TPAMI*），是我们先前工作 [BETA](https://github.com/DexterZeng/BETA) 的扩展。







