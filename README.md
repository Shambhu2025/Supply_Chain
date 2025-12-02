# SupplyChain-Graph-Hydra 
### Resilience Modeling with Multi-Task Graph Neural Networks (GNNs)

**Status:** Completed  
**Tech Stack:** PyTorch Geometric, NetworkX, Python

## 1. The Problem
Traditional supply chain analytics rely on linear regression or time-series forecasting. These methods fail to capture **systemic risk**—how a failure in one location (e.g., a port strike) propagates through hidden dependencies to affect downstream nodes.

## 2. The Solution: "The Hydra Architecture"
I built a custom Graph Neural Network (GNN) with a Multi-Task Learning architecture:
* **The Backbone:** A 2-layer **GraphSAGE** encoder that learns the topological structure of the supply network.
* **Head 1 (Link Prediction):** Predicts hidden or likely future supply routes using Dot Product similarity.
* **Head 2 (Node Regression):** Predicts operational risk (delay minutes) based on network centrality and neighbor states.

## 3. Key Result: The "Butterfly Effect"
Using Counterfactual Analysis (simulating a strike at Node 0), the model demonstrated that risk does not decay linearly with distance.
* **Direct Neighbor (Node 18):** Risk increased by **0.96**.
* **Downstream Node (Node 49):** Risk increased by **2.03**.

**Insight:** The model correctly identified that while Node 18 was closer to the failure, Node 49 was structurally more vulnerable (highly dependent on the specific pathway provided by Node 0).

![Impact Map](impact_map.png)

## 4. How to Run
1. Install dependencies: `pip install torch torch-geometric networkx`
2. Run the simulation: `python main.py`
3. The script will generate a synthetic supply chain, train the GNN, and output the "Strike Impact Report."

## 5. Future Scope
* Replacing synthetic data with the **SupplyGraph** benchmark dataset.
* Adding Temporal GNN layers (LSTM-GNN) to account for seasonality.
