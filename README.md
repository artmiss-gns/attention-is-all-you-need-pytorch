# Attention Is All You Need Pytorch Implementation

This repository contains a PyTorch implementation of the Transformer model, as described in the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. The goal of this project is to provide a clear and well-documented implementation for educational purposes.

The entire model is built from scratch using PyTorch

## Key Features

-   **Full Encoder-Decoder Architecture**: Implements the complete stack of encoders and decoders.
-   **Cross & Self-Attention**: A from-scratch implementation of the core attention mechanism.
p-   **Masking**: Properly implements source and target masks for training.

## Architecture Components

The following key components of the Transformer architecture are implemented:

-   Encoder Block
    -   Single-Head Attention
    -   Add & Norm (Layer Normalization)
    -   Feed-Forward Network
-   Decoder Block
    -   Masked Single-Head Attention
    -   Single-Head Cross-Attention (Encoder-Decoder)
    -   Add & Norm
    -   Feed-Forward Network
-   Input/Output Embeddings
-   Positional Encoding

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

-   Python 3.8+
-   A virtual environment tool (`venv` or `uv`)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/[your-github-username]/transformer-from-scratch.git
    cd transformer-from-scratch
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For venv
    python3 -m venv .venv
    source .venv/bin/activate

    # For uv
    uv venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The project is structured as a Jupyter Notebook (`main.ipynb`).

1.  **Launch Jupyter Lab:**
    ```sh
    jupyter lab
    ```

2.  **Run the notebook:**
    Open `main.ipynb` from the Jupyter interface and execute the cells sequentially. The notebook handles data loading, preprocessing, model training, and evaluation.

## Dataset

This model was trained on the **multi30k** dataset. The notebook includes all necessary steps to download and prepare the data for training.

## Acknowledgments

-   This implementation is heavily based on the original paper: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. "Attention Is All You Need."
