# Building LLMs from Scratch: A Hands-On Guide to GPT and Llama 3 Architectures

**Author:** Boufafa Mohamed  
**Tech Stack:** PyTorch | Python | Transformers | Deep Learning

---

This repository documents my journey into building Large Language Models (LLMs) from scratch. The goal is to demystify the core components of modern transformer architectures by implementing them step-by-step in PyTorch.

The repository is divided into two main sections:
1.  **A GPT-like Model**: We build a complete, character-level Generative Pre-trained Transformer from the ground up, starting with a simple bigram model and progressively adding complexity.
2.  **Llama 3 Architectural Deep Dive**: We explore the key innovations in Llama 3, by implementing its advanced attention, feed-forward, and tokenization mechanisms in focused notebooks.

## 📁 Repository Structure

```
LLM-From-Scratch/
│
├── GPT from Scratch/
│   ├── bigram.py               # Simple baseline bigram language model
│   ├── GPT.py                  # Full GPT-like transformer implementation
│   ├── GPT-Scratch.ipynb       # Step-by-step notebook with detailed explanations
│   └── TinyShakespear.txt      # Training dataset (Shakespeare text)
│
├── Llama 3 from Scratch/
│   ├── Llama_3_Attention.ipynb      # Advanced attention mechanism (RoPE, GQA)
│   ├── Llama_3_feed_forward.ipynb   # SwiGLU feed-forward network with RMSNorm
│   ├── BPE_Scratch.py               # Byte-Pair Encoding tokenizer script
│   └── BPE_Scratch.ipynb            # BPE implementation notebook
│
├── PaliGemma from scratch/
│   └── modeling_siglip.py      # SigLIP vision-language model implementation
│
├── .gitignore                  # Python/Jupyter gitignore configuration
└── README.md                   # Project documentation
```

---

## 🤖 Part 1: Building a GPT-like Model

This section focuses on constructing a decoder-only transformer model, similar in spirit to OpenAI's GPT-2, trained on the works of Shakespeare. It uses a simple **character-level tokenization** to demonstrate the core mechanics of the transformer architecture without the added complexity of a subword tokenizer.

### 🎯 Key Concepts Covered
-   Character-level Tokenization
-   Bigram Language Models
-   Transformer Architecture
-   Self-Attention Mechanism
-   Multi-Head Attention
-   Positional Embeddings
-   Layer Normalization and Residual Connections
-   Autoregressive Text Generation

### 📂 File Breakdown

#### 📋 `bigram.py`
This script serves as the "Hello World" of language modeling. It implements a simple Bigram model where the prediction for the next character depends only on the immediately preceding character. It's a crucial baseline that helps establish the data loading pipeline and evaluation framework before moving to a more complex architecture.

#### 🤖 `GPT.py` & `GPT-Scratch.ipynb`
These files are the core of this section.
-   The `GPT-Scratch.ipynb` notebook provides a narrative, cell-by-cell walkthrough of the entire process. It explains the "why" behind each component, from the mathematical trick in self-attention to the final model assembly.
-   The `GPT.py` script is a clean, runnable implementation of the final model.

The model built here includes all the fundamental components of a transformer block:
-   **Multi-Head Self-Attention**: Allows tokens to communicate with each other and weigh the importance of different tokens in the context.
-   **Feed-Forward Network**: A simple MLP applied to each token position independently, adding computational depth.
-   **Residual Connections & Layer Normalization**: Critical for stabilizing training in deep networks by preventing vanishing/exploding gradients.

---

## 🦙 Part 2: Deconstructing Llama 3's Architecture

This section moves beyond the foundational GPT architecture to explore the specific, high-performance components used in Meta's Llama 3 model. These notebooks isolate and implement these key innovations.

### 🎯 Key Concepts Covered
-   **Byte-Pair Encoding (BPE)**: A modern subword tokenization strategy.
-   **RMSNorm**: A simpler and more efficient alternative to LayerNorm.
-   **SwiGLU (Gated Linear Unit)**: An advanced feed-forward network that often provides better performance than standard ReLU-based networks.
-   **Rotary Positional Embeddings (RoPE)**: A sophisticated method for encoding positional information by rotating the query and key vectors.
-   **Grouped-Query Attention (GQA)**: An optimized attention mechanism balancing performance and efficiency.
-   **QK Normalization**: An additional normalization step to improve training stability.

### 📂 File Breakdown

#### 🧠 `Llama_3_Attention.ipynb`
This notebook is a deep dive into the heart of the Llama 3 transformer block: its attention mechanism. It meticulously reconstructs the entire attention forward pass, explaining and implementing RoPE, QK Norm, and Grouped-Query Attention (GQA).

#### 🚀 `Llama_3_feed_forward.ipynb`
This notebook focuses on the other major component of the Llama 3 transformer block: the feed-forward network. It implements the **SwiGLU** variant and demonstrates the use of **RMSNorm** for pre-normalization, a key feature of the Llama architecture.

#### 🧩 `BPE_Scratch.ipynb` & `BPE_Scratch.py`
A fundamental aspect of modern LLMs is how they process text. While our simple GPT model uses character-level tokens, advanced models like Llama 3 rely on **subword tokenization**. This notebook and script implement **Byte-Pair Encoding (BPE)** from scratch, a popular subword tokenization algorithm. It starts with a base vocabulary of individual characters and iteratively merges the most frequent adjacent pairs, creating a more efficient and semantically rich vocabulary. This is a crucial concept for understanding how LLMs handle vast and varied language data.

---

## 🚀 How to Use This Repository

### 📋 Prerequisites
-   Python 3.8+
-   PyTorch
-   Jupyter Notebook or JupyterLab (for `.ipynb` files)

### ⚙️ Setup and Running
1.  **Clone the repository:**
    ```bash
    git clone git@github.com:MohamedBoufafa/LLM-From-Scratch.git
    cd LLM-From-Scratch
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install torch torchvision torchaudio notebook
    ```

3.  **Download the dataset:**
    The `TinyShakespear.txt` dataset is used for the GPT model (if not already included).
    ```bash
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O "GPT from Scratch/TinyShakespear.txt"
    ```

4.  **Run the Python scripts:**
    To train the GPT model, navigate to the `GPT from Scratch` directory and run the script:
    ```bash
    cd "GPT from Scratch"
    python GPT.py
    ```

5.  **Explore the notebooks:**
    Launch Jupyter and navigate through the notebooks to see the step-by-step implementations.
    ```bash
    jupyter notebook
    ```

## 🙏 Acknowledgements
This project is heavily based on the excellent, educational content from online tutorials that break down these complex topics into manageable pieces.

Let's build GPT: from scratch, in code, spelled out. by Andrej Karpathy (https://youtu.be/kCc8FmEb1nY?feature=shared)

Code Llama 4 From Scratch - Easy Math Explanations & Python Code by Vuk Rosić (https://youtu.be/wcDV3l4CD14?feature=shared)

