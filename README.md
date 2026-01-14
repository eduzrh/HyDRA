<div align="center">

<h1>

✨ Towards Temporal Knowledge Graph Alignment in the Wild ✨

</h1>



<h3>—————— Under Review at IEEE TPAMI ——————</h3>

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
  <a href="#-introduction"><b>📰 Introduction</b></a> |
  <a href="#architecture"><b>🏗️ Architecture</b></a> |
  <a href="#installation"><b>⚙️ Installation</b></a> |
  <a href="#-quick-start"><b>🚀 Quick Start</b></a> <br>
  <a href="#-datasets"><b>📦 Datasets</b></a> |
  <a href="#-usage"><b>📖 Usage</b></a> |
  <a href="#-reproducibility"><b>🔬 Reproducibility</b></a> |
  <a href="#-license"><b>📜 License</b></a> |
  <a href="#-contact"><b>📬 Contact</b></a>
</p>



---

## 📰 Latest News

<div align="center">

| 🆕 Updates | 📅 Date | 📝 Description |
|:---:|:---:|:---|
| 🎉 **Code Release** | - | HyDRA codebase and datasets now available |

</div>

---

## 📰 Introduction



**Temporal Knowledge Graph Alignment in the Wild (TKGA-Wild)** addresses a critical challenge in temporal knowledge graph integration. To the best of our knowledge, this is the **first work** to formally formulate and solve this problem, which we term **TKGA-Wild**. This task presents unique challenges due to **Multi-Scale Temporal Elements** (i.e., multi-granular temporal coexistence and temporal span disparity) and **Asymmetric Temporal Structures** (i.e., heterogeneous temporal structures and temporal structural incompleteness) that are common in real-world scenarios.



To bridge this gap, we propose **HyDRA**, a new paradigm based on **multi-scale hypergraph retrieval-augmented generation** to systematically address the unique challenges of TKGA-Wild. HyDRA effectively captures complex structural dependencies, models multi-granular temporal features, mitigates temporal disparities, and introduces a new **scale-weave synergy mechanism** to coordinate information across different temporal scales.



## 🔥 Key Features



<div align="center">

| Feature | Icon | Description |
|:---|:---:|:---|
| **Multi-Granularity Temporal Encoding** | 🔄 | Captures temporal information at different scales (year, month, day) |
| **Scale-Adaptive Entity Projection** | 📐 | Adaptive entity projection across different graph scales and dimensions |
| **Multi-Scale Hypergraph Retrieval** | 🔍 | Efficient neural retrieval for hypergraph-based search |
| **Scale-Weave Synergy** | 🔗 | Coordinates information across different temporal scales |
| **State-of-the-Art Performance** | 📈 | Consistently outperforming 28 competitive baselines, achieving up to 43.3% improvement in Hits@1 |

</div>



---



## 🏗️ Architecture



HyDRA adopts a **multi-scale hypergraph retrieval-augmented generation** paradigm, comprising several key stages:


Stage 1: Encoding and Integration 🔄


Stage 2: Scale-Adaptive Entity Projection 📐

Stage 3: Multi-Scale Hypergraph Retrieval 🔍


Stage 4: Multi-Scale Fusion 🔗



> 📖 For detailed architecture descriptions and theoretical foundations, refer to the accompanying paper.



---



## ⚙️ Installation



### 📋 Prerequisites



First, install dependencies:



```bash

pip install -r requirements.txt

```



### 📦 Main Dependencies



| Package | Version | Purpose |
|:---|:---:|:---|
| 🐍 **Python** | >= 3.7 | Core language (tested on 3.8.10) |
| 🔥 **PyTorch** | >= 1.10.0 | Deep learning framework |
| 🔍 **Faiss** | >= 1.7.0 | Efficient similarity search (CPU/GPU) |
| 📊 **NumPy** | >= 1.21.0 | Numerical computing |
| 🐼 **Pandas** | >= 1.3.0 | Data manipulation |
| ⏳ **Tqdm** | >= 4.62.0 | Progress bars |
| 🌐 **NetworkX** | >= 2.6.0 | Graph analysis |



> 💡 **Note:** For GPU-accelerated FAISS, use `faiss-gpu` instead of `faiss-cpu`.



---



## 📦 Datasets



For our newly proposed **TKGA-Wild** scenario, we introduce two novel benchmark datasets: **BETA** and **WildBETA**.

<div align="center">

| Dataset | Description | Size |
|:---|:---|:---|
| **BETA** | Benchmark dataset for TKGA-Wild | - |
| **WildBETA** | Extended benchmark dataset for TKGA-Wild | - |

</div>

### 🔗 Download Links

<div align="center">

[![Baidu Netdisk](https://img.shields.io/badge/Baidu_Netdisk-Download-blue?style=for-the-badge)](https://pan.baidu.com/s/YOUR_LINK_HERE?pwd=YOUR_PWD)
[![Google Drive](https://img.shields.io/badge/Google_Drive-Download-green?style=for-the-badge)](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)

</div>

**Dataset Format:**



Take the dataset `icews_wiki` as an example, the folder `data/icews_wiki/` should contain:



- `ent_ids_1`: Entity IDs in source KG

- `ent_ids_2`: Entity IDs in target KG

- `triples_1`: Relation triples encoded by IDs in source KG

- `triples_2`: Relation triples encoded by IDs in target KG

- `rel_ids_1`: Relation IDs in the source KG

- `rel_ids_2`: Relation IDs in the target KG

- `time_id`: Time IDs in the source KG and the target KG

- `ref_ent_ids`: All aligned entity pairs, list of pairs like `(e_s \t e_t)`



**Note:** The standard temporal knowledge graph alignment datasets used in experiments are derived from [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [BETA](https://github.com/DexterZeng/BETA) and related works.



---



## 🚀 Quick Start



### Step 1: Clone the Repository 📥



```bash

git clone https://github.com/eduzrh/HyDRA.git

cd HyDRA

```



### Step 2: Prepare Datasets 📦



Download and extract datasets to `./data/`



### Step 3: Run the Main Experiment ▶️



```bash

python HyDRA_main.py --data_dir data/icews_wiki

```



This executes the complete **HyDRA** pipeline:



```

┌─────────────────────────────────────────┐

│  Encoding and Integration               │

│           ↓                              │

│  Scale-Adaptive Entity Projection       │

│           ↓                              │

│  Multi-Scale Hypergraph Retrieval       │

│           ↓                              │

│  Multi-Scale Fusion & Refinement        │

└─────────────────────────────────────────┘

```



### Step 4: View Results 📊



| Metric | Description |
|:---|:---|
| **Hits@1** | Proportion of correct alignments ranked first |
| **Hits@10** | Proportion in top-10 candidates |
| **MRR** | Mean Reciprocal Rank |



---



## 📖 Usage



### Basic Usage



**Run complete pipeline:**



```bash

python HyDRA_main.py --data_dir data/icews_wiki

```



**Skip encoding stage (if results already exist):**



```bash

python HyDRA_main.py --data_dir data/icews_wiki --skip_s4

```



**Run only encoding stage:**



```bash

python HyDRA_main.py --data_dir data/icews_wiki --only_s4

```



### Advanced Options



**Configure training parameters:**



```bash

python HyDRA_main.py --data_dir data/icews_wiki \

    --cuda 0 \

    --epochs 1500 \

    --max_iterations 5 \

    --min_kg1_entities 100

```



**Parameter Descriptions:**



| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--data_dir` | str | **Required** | Path to dataset directory |
| `--skip_s4` | flag | False | Skip encoding stage (if results already exist) |
| `--only_s4` | flag | False | Run only encoding stage |
| `--cuda` | int | 0 | CUDA device ID for training |
| `--epochs` | int | 500 | Number of training epochs for encoding stage |
| `--max_iterations` | int | 3 | Maximum pipeline iterations |
| `--min_kg1_entities` | int | 50 | Minimum entities threshold for stopping |



### Multi-Granularity Time Modeling



HyDRA supports multi-granularity temporal modeling (year and month levels) to handle Multi-Granular Temporal Coexistence. This feature can be enabled through the encoding stage configuration.



---



## 🔬 Reproducibility



We are committed to ensuring full reproducibility of our results. The following resources are provided:



### 📋 Experimental Configuration



- **Hyperparameters**: All hyperparameter settings are documented in the code and can be configured via command-line arguments

- **Random Seeds**: Seed configurations are embedded in the training scripts for reproducibility

- **Environment**: Tested on Python 3.8.10 with dependencies as specified in `requirements.txt`



### 📊 Reproducing Main Results



To reproduce the main experimental results reported in the paper:



1. **Download datasets** following the format described in the Datasets section

2. **Run the complete pipeline** with default settings:



```bash

python HyDRA_main.py --data_dir data/icews_wiki

```



3. **Evaluate results** using the output files in `data/icews_wiki/message_pool/`



### 🏗️ Code Organization



The codebase is organized into modular components for clarity:



- `encoding_and_integration/`: Multi-granularity temporal entity encoding and integration

- `scale_adaptive_entity_projection/`: Relation alignment and entity projection

- `multi_scale_hypergraph_retrieval/`: Neural retrieval and hypergraph decomposition

- `multi_scale_fusion/`: Multi-scale fusion and alignment refinement

- `HyDRA_main.py`: Main pipeline orchestrator



### 📝 Documentation



- Comprehensive inline code comments explaining key design decisions

- Clear module structure with standardized naming conventions

- This README with step-by-step usage instructions



---



## 📊 Evaluation Metrics



We employ standard knowledge graph alignment metrics for transparency and comparability:



- **Hits@1**: Proportion of correct alignments ranked first

- **Hits@10**: Proportion of correct alignments in top-10 candidates  

- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct alignments



## 📬 Contact



- **Email**: [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)

- **GitHub Issues**: For technical concerns, create an Issue in the [GitHub repository](https://github.com/eduzrh/HyDRA/issues). Labels: `bug`, `enhancement`, `question`.



Responses targeted within 2-3 business days.



## 📜 License



[MIT License](LICENSE) - Copyright notices preserved.



---




## 🔗 References

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



## 🙏 Acknowledgement

The following open source projects were partially referenced in this work. We sincerely appreciate their contributions:

[Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [Simple-HHEA](https://github.com/IDEA-FinAI/Simple-HHEA), [BETA](https://github.com/DexterZeng/BETA), [Dual-Match](https://github.com/ZJU-DAILY/DualMatch/), [Faiss](https://github.com/facebookresearch/faiss), [NetworkX](https://github.com/networkx/networkx), [AdaCoAgentEA](https://github.com/eduzrh/AdaCoAgentEA)

---

This repository corresponds to the paper ***Towards Temporal Knowledge Graph Alignment in the Wild*** (under review at *IEEE TPAMI*), and is an extension of our previous work [BETA](https://github.com/DexterZeng/BETA).
