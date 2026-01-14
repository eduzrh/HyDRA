<div align="center">

<h1>

🔬 HyDRA: Multi-Scale Hypergraph Neural Retrieval and Alignment 🔬

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

## 📰 Introduction

**HyDRA (Hypergraph Neural Retrieval and Alignment)** addresses the challenge of **Temporal Knowledge Graph Alignment in the Wild (TKGA-Wild)**, which systematically handles **Multi-Scale Temporal Elements** and **Asymmetric Temporal Structures** in real-world temporal knowledge graphs. The framework introduces a novel paradigm based on **multi-scale hypergraph retrieval-augmented generation** to effectively capture complex structural dependencies, model multi-granular temporal features, and mitigate temporal disparities.

The core innovation of HyDRA lies in its ability to handle:

- **Multi-Scale Temporal Elements**: Multi-Granular Temporal Coexistence (year, month, day) and Temporal Span Disparity
- **Asymmetric Temporal Structures**: Heterogeneous Temporal Structures and Temporal Structural Incompleteness
- **Multi-scale hypergraph retrieval-augmented generation**: Systematic integration of retrieval and generation mechanisms
- **Scale-weave synergy**: Coordinating information across different temporal scales

## 🔥 Key Features

<div align="center">

| Feature | Icon | Description |

|:---|:---:|:---|

| **Encoding & Integration** | 🔄 | Multi-granularity temporal entity encoding and integration |

| **Scale-Adaptive Entity Projection** | 📐 | Adaptive entity projection across different graph scales and dimensions |

| **Multi-Scale Hypergraph Retrieval** | 🔍 | Efficient neural retrieval using FAISS for hypergraph-based similarity search |

| **Multi-Scale Fusion** | 🔗 | Iterative alignment refinement through multi-scale information fusion |

| **Reproducibility** | ✅ | Comprehensive documentation and configuration files for full reproducibility |

</div>

---

## 🏗️ Architecture

HyDRA adopts a **multi-scale hypergraph retrieval-augmented generation** paradigm, comprising several key stages:

**Stage 1: Encoding and Integration** 🔄

- Performs multi-granularity temporal entity encoding to capture temporal information at different scales (year, month, day)
- Generates initial entity similarity scores through neural embedding learning
- Integrates temporal features to handle Multi-Granular Temporal Coexistence

**Stage 2: Scale-Adaptive Entity Projection** 📐

- Executes relation alignment to identify corresponding relations across temporal knowledge graphs
- Performs hypergraph decomposition to extract multi-scale representations
- Adaptively projects entities across different scales to handle Heterogeneous Temporal Structures

**Stage 3: Multi-Scale Hypergraph Retrieval** 🔍

- Conducts efficient neural retrieval using FAISS for hypergraph-based similarity search
- Builds multi-scale hypergraph representations for capturing complex structural dependencies
- Handles Temporal Span Disparity through scale-adaptive retrieval mechanisms

**Stage 4: Multi-Scale Fusion** 🔗

- Applies multi-scale fusion to combine alignment signals from different temporal scales
- Implements scale-weave synergy to coordinate information across scales
- Iteratively refines alignments and updates seed entity pairs

> 📖 For detailed architecture descriptions and theoretical foundations, refer to the accompanying paper.

---

## ⚙️ Installation

### 📋 Prerequisites

- Python >= 3.7 (tested on Python 3.8.10)
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended for large datasets)

### 📦 Installation Steps

1. **Clone the repository:**

```bash

git clone https://github.com/eduzrh/HyDRA.git

cd HyDRA

```

2. **Install dependencies:**

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

### 🔑 Configure Environment (Optional)

For LLM-based fusion components (if used), configure API credentials:

```env

LLM_API_KEY=your_key_here

LLM_API_BASE=your_base_here

```

---

## 📦 Datasets

HyDRA supports standard temporal knowledge graph alignment datasets. The expected dataset format follows the standard structure for temporal knowledge graph alignment tasks.

**Dataset Structure:**

For a dataset (e.g., `icews_wiki`), the folder `data/icews_wiki/` should contain:

- `ent_ids_1`: Entity IDs in source knowledge graph

- `ent_ids_2`: Entity IDs in target knowledge graph

- `triples_1`: Temporal relation triples in source KG (format: `head \t relation \t tail \t time_start \t time_end`)

- `triples_2`: Temporal relation triples in target KG

- `rel_ids_1`: Relation IDs in source KG

- `rel_ids_2`: Relation IDs in target KG

- `time_id`: Time ID mappings (format: `time_id \t time_string`)

- `ref_ent_ids`: Reference entity alignments (format: `entity_1 \t entity_2`)

**Dataset Sources:**

The datasets used in our experiments are derived from publicly available sources:

- Standard temporal KG alignment datasets
- [Dual-AMN](https://github.com/MaoXinn/Dual-AMN) - Entity alignment benchmarks
- [JAPE](https://github.com/nju-websoft/JAPE) - Cross-lingual KG alignment
- [BETA](https://github.com/DexterZeng/BETA) - Temporal KG embedding

> 📝 Please refer to the respective repositories for dataset download links and licensing information.

---

## 🚀 Quick Start

### Step 1: Prepare Dataset 📦

Ensure your dataset follows the structure described above and is placed in the `data/` directory.

```bash

# Example structure

data/

  └── icews_wiki/

      ├── ent_ids_1

      ├── ent_ids_2

      ├── triples_1

      ├── triples_2

      ├── rel_ids_1

      ├── rel_ids_2

      ├── time_id

      └── ref_ent_ids

```

### Step 2: Run the Complete Pipeline ▶️

Execute the full HyDRA pipeline:

```bash

python HyDRA_main.py --data_dir data/icews_wiki

```

This executes the complete pipeline:

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

### Step 3: View Results 📊

Results are saved in the `data/icews_wiki/message_pool/` directory:

- `integration_top_pair.txt`: Initial entity alignments from encoding stage
- `retriever_outputs.txt`: Neural retrieval results
- Additional intermediate outputs for each stage

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

HyDRA uses standard knowledge graph alignment evaluation metrics:

- **Hits@1**: Proportion of correct alignments ranked first
- **Hits@10**: Proportion of correct alignments in top-10 candidates  
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct alignments

Results are computed based on the output files generated by the pipeline.

---

## 📬 Contact

- **Email**: [Your Email](mailto:your.email@example.com)
- **GitHub Issues**: For technical questions or bug reports, please create an Issue in the [GitHub repository](https://github.com/eduzrh/HyDRA/issues)

We aim to respond within 2-3 business days.

---

## 📜 License

[MIT License](LICENSE) - See LICENSE file for details.

---

## 🔗 References

We acknowledge the following open-source projects and datasets that were referenced or used in this work:

- Temporal knowledge graph alignment frameworks and datasets
- [Dual-AMN](https://github.com/MaoXinn/Dual-AMN) - Entity alignment with attention matching
- [JAPE](https://github.com/nju-websoft/JAPE) - Cross-lingual knowledge graph alignment
- [GCN-Align](https://github.com/1049451037/GCN-Align) - Graph convolutional networks for alignment
- [BETA](https://github.com/DexterZeng/BETA) - Temporal knowledge graph embedding
- [Faiss](https://github.com/facebookresearch/faiss) - Efficient similarity search library
- [NetworkX](https://github.com/networkx/networkx) - Network analysis in Python

---

## 🙏 Acknowledgement

We sincerely appreciate the contributions of the open-source community and the developers of the referenced projects. Their work has been instrumental in advancing knowledge graph alignment research.

---

<div align="center">

**⭐ If you find this work useful, please consider starring the repository! ⭐**

</div>
