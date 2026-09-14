# 🧬 De Novo Molecular Generation Using LSTM-Based VAE

[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Chemoinformatics-0284C7)](https://www.rdkit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Chemical Validity](https://img.shields.io/badge/Chemical%20Validity-99.02%25-059669)](#-results)
[![Lipinski Compliance](https://img.shields.io/badge/Lipinski%20Compliance-99.98%25-059669)](#-results)

An end-to-end deep learning framework for **de novo molecular generation** using an **LSTM-based Variational Autoencoder (VAE) with a Bidirectional LSTM Encoder**, trained on the **ZINC250K** molecular dataset.

The framework learns molecular representations from SMILES sequences and generates novel molecular structures through stochastic latent-space sampling and autoregressive decoding. The proposed architecture incorporates **step-wise latent injection, auxiliary molecular-property prediction, and β-KL annealing** to improve latent representation learning and molecular generation.

The project also includes an interactive **Streamlit Web Platform** (`app.py`) for generating molecular structures, visualizing their 2D representations, applying Lipinski's Rule of Five filtering, inspecting molecular properties, and exporting generated results.

---

## 🔬 Project Overview

The project focuses on computational **de novo molecular generation**, where a deep generative model learns structural patterns from existing molecules and generates new molecular structures represented as SMILES strings.

The model is trained on the **ZINC250K dataset containing 249,455 molecules**. SMILES representations are validated and canonicalized using RDKit, tokenized using a regex-based tokenizer, and processed for sequence-based deep learning.

The proposed model consists of:

- **Bidirectional LSTM Encoder**
- **128-dimensional latent representation**
- **LSTM Decoder**
- **Step-wise latent injection**
- **Auxiliary molecular-property prediction**
- **β-KL annealing**
- **Temperature-controlled Top-p sampling**

The generated molecules are evaluated using chemical validity, uniqueness, novelty, internal diversity, Lipinski compliance, QED, and synthetic accessibility.

---

## 🧠 Key Architectural Components

### 1. Bidirectional LSTM Encoder

A two-layer Bidirectional LSTM encoder processes the SMILES sequence in both forward and backward directions.

Each molecular token is represented using a **256-dimensional embedding**, with a hidden dimension of 256. The forward and backward representations are combined and projected into a **128-dimensional latent distribution** represented by its mean and log variance.

### 2. Latent-Space Representation

The encoder produces the parameters of the molecular latent distribution. During training, the latent vector is sampled using the **reparameterization trick**.

This latent representation captures molecular information and forms the connection between the encoder and decoder.

### 3. Step-wise Latent Injection

The proposed decoder uses **step-wise latent injection**, where the latent vector is concatenated with the token embedding at every decoding step.

This allows the decoder to continuously access the learned latent representation during SMILES reconstruction and generation.

### 4. Auxiliary Molecular-Property Prediction

An auxiliary MLP predicts selected molecular properties directly from the latent representation:

- **LogP**
- **QED**
- **Synthetic Accessibility (SA) Score**

This additional prediction objective encourages the latent representation to retain chemically relevant molecular information.

### 5. β-KL Annealing

The training objective combines:

- SMILES Reconstruction Loss
- KL-Divergence Loss
- Molecular Property Prediction Loss

The β coefficient for KL regularization is gradually increased during the initial training stage until reaching a maximum value of **0.05**, helping balance reconstruction learning and latent-space regularization.

### 6. Temperature + Top-p Sampling

During molecular generation:

- **Temperature = 0.8**
- **Top-p = 0.9**

These sampling strategies control the stochasticity of autoregressive SMILES generation and promote molecular diversity.

---

## 📊 Results

The trained model was used to generate **10,000 SMILES sequences**.

| Metric | Result |
|---|---:|
| **Molecules Generated** | 10,000 |
| **Chemical Validity** | **99.02%** |
| **Uniqueness** | **99.75%** |
| **Exact-Match Novelty** | **99.65%** |
| **Internal Diversity** | **84.57%** |
| **Lipinski Compliance*** | **99.98%** |
| **Mean QED Score** | **0.7719** |
| **Mean SA Score** | **2.63** |

\*Lipinski compliance is calculated with respect to the 9,902 chemically valid molecules.

### Generation Performance

Out of 10,000 generated SMILES sequences:

- **9,902** were chemically valid.
- **9,877** of the valid molecules were unique.
- **99.65%** showed exact-match novelty against the ZINC250K reference set.
- **84.57%** internal diversity was achieved.
- **9,900 / 9,902** valid molecules satisfied Lipinski's Rule of Five.

The generated molecular set achieved a mean **QED score of 0.7719** and a mean **SA score of 2.63**.

---

## 📈 Training Performance

The model was trained for **15 epochs** using:

| Parameter | Value |
|---|---|
| Dataset | ZINC250K |
| Training Molecules | 224,509 |
| Validation Molecules | 24,946 |
| Vocabulary Size | 66 tokens |
| Embedding Dimension | 256 |
| Hidden Dimension | 256 |
| Latent Dimension | 128 |
| LSTM Layers | 2 |
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Maximum β | 0.05 |

During training:

- Training loss decreased from **0.9644 → 0.6604**
- Validation loss decreased from **0.7097 → 0.6331**
- Reconstruction loss decreased from **0.9017 → 0.5899**
- KL loss decreased from **4.5323 → 0.7476**

The lowest validation loss was **0.6331 at Epoch 15**.

---

## 🧪 Molecular Evaluation

Generated molecules are evaluated using multiple complementary metrics.

### Chemical Validity
RDKit is used to determine whether generated SMILES strings correspond to chemically valid molecular structures.

### Uniqueness
Canonical SMILES representations are used to identify duplicate generated structures.

### Novelty
Exact-match novelty is evaluated using molecular **InChIKeys** against reference molecular databases.

### Internal Diversity
Morgan fingerprints (**ECFP4, radius 2, 2048 bits**) and pairwise Tanimoto similarity are used to measure structural diversity.

### Drug-Likeness
Generated molecules are evaluated using:

- **Lipinski's Rule of Five**
- **QED Score**
- **Synthetic Accessibility (SA) Score**

---

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit interface for molecular generation.

### Features

- 🎛️ Number of molecules/candidates control
- 🌡️ Temperature sampling control
- 🎯 Top-p (nucleus) sampling
- 🧪 Lipinski's Rule of Five filtering
- 🧬 SMILES generation
- 🔬 2D molecular structure visualization
- 📊 Molecular property profiling
- 📁 CSV result export

### Run Locally

```bash
streamlit run app.py
