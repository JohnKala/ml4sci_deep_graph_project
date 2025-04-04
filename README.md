# Quark/Gluon Classification with Graph Contrastive Learning

This project implements a Graph Neural Network with contrastive learning to classify quark/gluon jets. The approach transforms jet images into point clouds and graph structures, then applies contrastive learning techniques to learn better representations of the data.

## Overview

Contrastive learning is a self-supervised technique that helps models learn meaningful representations by bringing similar samples closer together in embedding space while pushing dissimilar samples apart. In this project, we apply this approach to graph-structured jet data to improve classification performance.

## Implementation Details

### Data Processing Pipeline
1. Load quark/gluon jet images (ECAL, HCAL, and Tracks channels)
2. Convert images to point clouds by extracting non-zero pixels
3. Transform point clouds into graph structures with k-nearest neighbors
4. Create data augmentations via feature masking and edge dropout

### Contrastive Learning Model
- **Graph Encoder**: Multi-layer GCN network that extracts features from the graph structure
- **Projection Head**: MLP that projects embeddings to the space where contrastive loss is applied
- **Classifier**: Linear layer for the final quark/gluon classification
- **NT-Xent Loss**: Normalized temperature-scaled cross entropy loss for contrastive learning

### Training Approach
- Combined loss function that balances contrastive learning and classification objectives
- Multiple data augmentations for each graph to enable contrastive learning
- Learning rate scheduling and early stopping for optimal convergence
- Model checkpointing to save the best-performing version

### Evaluation
- Comprehensive metrics calculation: accuracy, precision, recall, F1 score, AUC
- Comparison with a baseline GNN without contrastive learning
- Embedding visualization with t-SNE to assess learned representations
- Detailed analysis of model performance

## How to Run

1. Ensure all dependencies are installed:
  ```bash
  pip install torch torch-geometric numpy matplotlib scikit-learn tqdm h5py scipy
