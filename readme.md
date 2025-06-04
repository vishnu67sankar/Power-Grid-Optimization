# Transfer Learning for Electric Power Grid Simulations
This repo is a lightweight deep learning model, that is easy to install and understand without getting lost in other details. It uses Graph Neural Networks (GNNs) and Transformer-based models, to predict voltage and phases of electric power grids.

The project uses PyTorch and PyTorch Geometric for building and training neural network models, and Pandapower for power system data simulation and manipulation.

This project was inspired by the following projects: https://github.com/Amirtalebi83/GNN-OptimalPowerFlow/tree/main and https://github.com/mukhlishga/gnn-powerflow/tree/main which discuss about general graph neural network approaches for the electric power grid problem.

## Project Structure
<pre lang="markdown">
Power Grid Optimization/
├── powergrid_torch/
│   ├── Data/              
│   │   └── 14Bus/..xlsx
│   │   └── 30Bus/..xlsx
│   │   └── 57Bus/..xlsx
│   │   └── 118Bus/..xlsx
│   ├── JupyterNotebooks/              
│   │   └── Model_Comparison.ipynb
│   ├── models/                        
│   │   ├── baseClass.py               
│   │   ├── gnn.py                     
│   │   ├── transformer_pg_model.py    
│   │   └── graph_transformer_model.py 
│   ├── training/                      
│   │   └── train.py
│   ├── utils/                         
│   │   ├── pre_processing.py          
│   │   └── __init__.py
├── powergrid_torch.egg-info/         
├── Readme.md                         
└── setup.py                          
</pre>


## Features

* **Multiple GNN Architectures**: Implements various GNN layers (e.g., GCN, GraphConv, SAGEConv, GATConv, ChebConv, TransformerConv)
* **Transformer Models**: Explores Graph Transformer
* **Data Pre-processing**: Utilities for creating datasets from power grid data, suitable for GNN and Transformer model training
* **Modular Training Framework**: A base model class (`PowerGridModelBase`) with standardized training and validation steps.
* **Transfer Learning Capabilities**: Still a work in progress, to be designed with considerations for applying pre-trained models to different grid configurations (e.g. varying number of buses)

## Dependencies

The primary dependencies for this project include:

* Python (3.9+)
* PyTorch (2.6.0)
* PyTorch Geometric (2.6.1)
* Pandapower
* NumPy
* Matplotlib 

**Installation Steps:**

1.  **PyTorch:**
    Follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) for your specific OS and CUDA version (if using GPU)

2.  **PyTorch Geometric:**
    Follow the instructions on the [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), make sure it is compatibile with your PyTorch version

3.  **Other Dependencies:**
    ```
    pip install pandapower numpy matplotlib plotly torchinfo jupyterlab
    ```

## Models

This project explores several types of neural network architectures:

### 1. Graph Neural Networks (`gnn.py`)

* Implements various GNN layers from PyTorch Geometric (GCN, GraphConv, SAGE, GAT, ChebConv)
* The final linear layer's size is dependent on `n_bus`, allowing for adaptation to different grid sizes (potentially with transfer learning)

### 2. Graph Transformer (`graph_transformer_model.py`)

* Utilizes `TransformerConv` from PyTorch Geometric
* Directly applies attention mechanisms over the graph structure

## Transfer Learning

The project includes considerations for transfer learning, particularly for the `GNNModel`. The idea is to pre-train a model on one grid configuration (e.g., a 14-bus system) and then transfer its learned weights (excluding the final, size-dependent layer) to a new model for a different grid size (e.g., a 40-bus system). This can speed up training and improve performance on the new task.

## Preliminary Results
This plot compares training and validation loss between `GraphConv` and `GraphTransformer` models over 400 epochs.

![Training Loss Plot](/powergrid_torch/results/model_comparison.png)