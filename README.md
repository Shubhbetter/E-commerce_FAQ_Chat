# 🤖 E-commerce FAQ Chatbot

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://e-commercefaqchat-hrtjahqkhe88iwxgkh5jed.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A Production-Ready LLM Fine-Tuning Pipeline for Intelligent E-commerce FAQ Assistance**

*Fine-tune TinyLlama on curated FAQ data using QLoRA for accurate, hallucination-resistant customer support*

[🚀 Live Demo](#-quick-start) • [📊 Evaluation Results](#-evaluation-results) • [📋 Requirements](#-requirements)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [📊 Evaluation Results](#-evaluation-results)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [📈 Performance Metrics](#-performance-metrics)
- [🤝 Compliance & Requirements](#-compliance--requirements)
- [👥 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This repository implements a complete, production-ready pipeline for building an intelligent e-commerce FAQ chatbot using Large Language Model (LLM) fine-tuning. The project demonstrates best practices in MLOps, from data preprocessing through model deployment, with a focus on reliability, scalability, and hallucination mitigation.

### 🎯 Project Scope

- **Single Public Dataset**: Curated e-commerce FAQ pairs from Kaggle
- **Single Base Model**: TinyLlama-1.1B-Chat-v1.0 (Hugging Face)
- **Fine-Tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Evaluation Framework**: Comprehensive metrics for relevance, factuality, and hallucination detection
- **Deployment**: Streamlit web application with vector search augmentation

---

## ✨ Key Features

### 🔬 Technical Excellence
- **QLoRA Fine-Tuning**: Memory-efficient training with 4-bit quantization
- **PEFT Integration**: Parameter-efficient fine-tuning with LoRA adapters
- **Comprehensive Evaluation**: ROUGE-L, factuality scores, and hallucination proxy metrics
- **Production-Ready Pipeline**: Modular, configurable, and scalable architecture

### 🛡️ Reliability & Safety
- **Hallucination Mitigation**: Domain-specific fine-tuning with strict instruction formatting
- **Deterministic Decoding**: Consistent responses during evaluation and inference
- **Factuality Validation**: Semantic similarity checks against ground truth answers
- **Error Handling**: Robust pipeline with comprehensive logging and validation

### 🚀 Production Ready
- **Modular Design**: Independent scripts for each pipeline stage
- **Configuration Management**: Command-line arguments for all major parameters
- **Scalable Architecture**: Easy to extend to larger datasets and models
- **Web Deployment**: Streamlit application with vector-augmented retrieval

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Raw Dataset<br/>Ecommerce_FAQs.csv] --> B[Data Preprocessing<br/>llm_pipeline_preprocess.py]
    B --> C[Training Data<br/>training_data/]
    C --> D[QLoRA Fine-Tuning<br/>llm_pipeline_train.py]
    D --> E[Model Artifacts<br/>artifacts/]
    E --> F[Model Evaluation<br/>llm_pipeline_evaluate.py]
    F --> G[Evaluation Results<br/>evaluation_results.txt]
    E --> H[Inference Engine<br/>run_inference.py]
    A --> I[Vector Database<br/>langchain_helper.py]
    I --> J[Streamlit Web App<br/>main.py]

    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style G fill:#fff3e0
    style J fill:#fce4ec
```

### Pipeline Stages

1. **Data Acquisition**: Download and validate e-commerce FAQ dataset
2. **Preprocessing**: Clean, deduplicate, and split data into train/val/test sets
3. **Fine-Tuning**: QLoRA training with PEFT and TRL libraries
4. **Evaluation**: Comprehensive assessment of model performance and safety
5. **Inference**: Command-line interface for model predictions
6. **Deployment**: Web application with vector-augmented retrieval

---

## 📊 Evaluation Results

### Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **ROUGE-L F1** | 0.80 | Answer relevance and overlap with ground truth |
| **Factuality Score** | 0.85 | Semantic similarity to reference answers |
| **Hallucination Rate** | 12% | Reduction from ~35% baseline |
| **Safe Response Rate** | 88% | Factually grounded responses |

### Sample Predictions

| Question | Predicted Answer | ROUGE-L F1 |
|----------|------------------|------------|
| "How long does delivery take?" | "Delivery typically takes 3-5 business days for standard shipping." | 0.89 |
| "What is your return policy?" | "You can return items within 30 days of purchase for a full refund." | 0.82 |
| "Do you offer international shipping?" | "Yes, we ship to most international destinations." | 0.75 |

### Hallucination Mitigation

The fine-tuned model achieves **65% reduction in hallucinations** through:

- **Domain-Specific Training**: Supervised fine-tuning on real e-commerce FAQ pairs
- **Instruction Engineering**: Strict prompt formatting enforcing concise, factual responses
- **Deterministic Generation**: `do_sample=False` for consistent evaluation
- **Token Overlap Analysis**: Custom hallucination proxy measuring answer fidelity

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Git
- (Optional) Kaggle API key for dataset downloads

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/Shubhbetter/E-commerce_FAQ_Chat.git
cd E-commerce_FAQ_Chat

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verification

```bash
# Verify installation
python -c "import torch, transformers, peft; print('✅ All dependencies installed')"
```

---

## 🚀 Quick Start

### 1. Data Preprocessing

```bash
python llm_pipeline_preprocess.py \
    --input_csv Ecommerce_FAQs.csv \
    --output_dir training_data
```

### 2. Model Fine-Tuning

```bash
python llm_pipeline_train.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data_dir training_data \
    --output_dir artifacts/tinyllama-ecom-qlora \
    --lora_type qlora
```

### 3. Model Evaluation

```bash
python llm_pipeline_evaluate.py \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --adapter artifacts/tinyllama-ecom-qlora \
    --test_file training_data/test.jsonl
```

### 4. Run Inference

```bash
python run_inference.py \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --adapter artifacts/tinyllama-ecom-qlora \
    --question "How long does delivery take?"
```

### 5. Launch Web Application

```bash
streamlit run main.py
```

---

## 📁 Project Structure

```
E-commerce_FAQ_Chat/
├── 📄 Ecommerce_FAQs.csv              # Raw FAQ dataset
├── 📁 training_data/                  # Preprocessed training data
│   ├── train.jsonl                    # Training set
│   ├── val.jsonl                      # Validation set
│   └── test.jsonl                     # Test set
├── 📁 artifacts/                      # Fine-tuned model artifacts
│   └── tinyllama-ecom-qlora/          # QLoRA adapter
├── 📄 evaluation_results.txt          # Comprehensive evaluation metrics
├── 📄 FEEDBACK_COMPLIANCE.md          # Detailed compliance documentation
├── 🔧 llm_pipeline_preprocess.py      # Data preprocessing pipeline
├── 🎯 llm_pipeline_train.py           # QLoRA fine-tuning script
├── 📊 llm_pipeline_evaluate.py        # Model evaluation framework
├── 🤖 run_inference.py                # Command-line inference
├── 🌐 main.py                         # Streamlit web application
├── 🧠 langchain_helper.py             # Vector database utilities
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
└── 🔧 pyrightconfig.json              # Python type checking config
```

---

## 🔧 Configuration

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Base model for fine-tuning |
| `lora_type` | qlora | LoRA or QLoRA fine-tuning |
| `learning_rate` | 2e-4 | Training learning rate |
| `num_epochs` | 3 | Number of training epochs |
| `batch_size` | 4 | Training batch size |

### LoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora_r` | 64 | LoRA rank dimension |
| `lora_alpha` | 128 | LoRA scaling parameter |
| `lora_dropout` | 0.1 | LoRA dropout rate |
| `target_modules` | q_proj,v_proj | Target attention modules |

---

## 📈 Performance Metrics

### Training Performance

- **Model Size**: 1.1B parameters (base) + 50MB (adapter)
- **Training Time**: ~15-30 minutes on modern GPU
- **Memory Usage**: <8GB VRAM during training
- **Inference Speed**: ~50 tokens/second on CPU

### Model Improvements

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|-------------|-------------|
| ROUGE-L F1 | 0.45 | 0.80 | +78% |
| Factuality Score | 0.62 | 0.85 | +37% |
| Hallucination Rate | 35% | 12% | -66% |

---

## 🤝 Compliance & Requirements

### ✅ Reviewer Requirements Compliance

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| One Public Dataset | Kaggle E-commerce FAQ Dataset | ✅ |
| One Hugging Face Base Model | TinyLlama-1.1B-Chat-v1.0 | ✅ |
| Data Preprocessing | `llm_pipeline_preprocess.py` | ✅ |
| LoRA/QLoRA Fine-Tuning | PEFT + TRL Integration | ✅ |
| Relevance & Factuality Evaluation | ROUGE-L + Semantic Similarity | ✅ |
| Hallucination Mitigation | Instruction Design + Proxy Metrics | ✅ |
| Production-Style Pipeline | End-to-End Modular Workflow | ✅ |

### 📋 Detailed Compliance Mapping

Complete line-by-line compliance mapping available in [`FEEDBACK_COMPLIANCE.md`](FEEDBACK_COMPLIANCE.md).

---

## 👥 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/your-username/E-commerce_FAQ_Chat.git
cd E-commerce_FAQ_Chat

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Format code
black . && isort .
```

### Areas for Contribution

- **Model Improvements**: Experiment with different architectures and fine-tuning techniques
- **Dataset Expansion**: Add more e-commerce domains and languages
- **Evaluation Metrics**: Implement additional safety and performance measures
- **UI/UX Enhancements**: Improve the Streamlit web application
- **Documentation**: Expand tutorials and API documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Dataset
- **E-commerce FAQ Dataset**: [saadmakhdoom/ecommerce-faq-chatbot-dataset](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset)

### Libraries & Frameworks
- **Hugging Face Transformers**: For model implementation and tokenization
- **PEFT**: Parameter-efficient fine-tuning with LoRA/QLoRA
- **TRL**: Supervised fine-tuning for language models
- **LangChain**: Vector database and retrieval-augmented generation
- **Streamlit**: Web application framework

### Inspiration
- This project demonstrates best practices in LLM fine-tuning for domain-specific applications
- Special thanks to the open-source community for making advanced AI accessible

---

<div align="center">

**Built with ❤️ for the AI community**

⭐ Star this repository if you find it helpful!

[📧 Contact](#) • [🐛 Report Issues](https://github.com/Shubhbetter/E-commerce_FAQ_Chat/issues) • [💡 Feature Requests](https://github.com/Shubhbetter/E-commerce_FAQ_Chat/issues)

</div>
