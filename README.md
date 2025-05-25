# **Reproducibility Study: Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics**

This repository contains a Python implementation reproducing the paper:  
**"Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics"** by Ofir Maron and Benjamin Rosman (2020).  

[Original Paper](https://www.raillab.org/publication/marom-2020-utilising/) | [Original C# Code](https://github.com/OfirMaron/LearnHeuristicWithUncertainty)

## **📋 Table of Contents**
1. [Installation](#-installation)
2. [Running Experiments](#-running-the-experiments)
3. [Code Structure](#-repository-structure)
4. [Results Interpretation](#-results-interpretation)
5. [Debugging Tips](#-debugging-tips)
6. [License](#-license)


## ** Installation**

### **Prerequisites**
- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended) or `pip`

### **Setup**
```bash
git clone https://github.com/your-username/reproducibility-assignment.git
cd reproducibility-assignment

# With Poetry (recommended):
poetry install

# With pip:
pip install -r requirements.txt
```





## **🏃 Running the Experiments**
1. Suboptimality Experiment

``` bash
python suboptimality_experiment.py
```

### ** What it does:***

Trains a Weight Uncertainty Neural Network (WUNN) and Feedforward Neural Network (FFNN)

Evaluates heuristic quality on 15-puzzle benchmarks


you will get sets of logs one fo wunn and one for FFnn

for ffnn 

smaple output 

State MD: 5
NN pred - mean: 0.18, σ_a: 3.74, σ_e: 0.24
Using FFNN prediction: -13.07
Final heuristic: 4.50 (MD: 5)


### **Log Interpretation:**

Term	Meaning
State MD	Manhattan distance (optimal baseline)
NN pred - mean	Predicted cost-to-goal
σ_a	Aleatoric uncertainty (data noise)
σ_e	Epistemic uncertainty (model uncertainty)
FFNN prediction	Raw neural network output
Final heuristic	Adjusted heuristic value used by planner



for wunn we get two one is the warmup and he other is teh actualt traiingin it forms part of th erun curiculma as that is how we make use of the learn heuristics

[Warmup] Epoch 60/100 - Loss: 3.5068
[Warmup] Epoch 61/100 - Loss: 3.6775
[Warmup] Epoch 62/100 - Loss: 3.0600
[Warmup] Epoch 63/100 - Loss: 3.2072
[Warmup] Epoch 64/100 - Loss: 2.9231
[Warmup] Epoch 65/100 - Loss: 2.6207
[Warmup] Epoch 66/100 - Loss: 2.4160
[Warmup] Epoch 67/100 - Loss: 2.5377
[Warmup] Epoch 68/100 - Loss: 2.2515
[Warmup] Epoch 69/100 - Loss: 2.2852
[Warmup] Epoch 70/100 - Loss: 2.0564
[Warmup] Epoch 71/100 - Loss: 2.1187
[Warmup] Epoch 72/100 - Loss: 1.9868
[Warmup] Epoch 73/100 - Loss: 2.1216
[Warmup] Epoch 74/100 - Loss: 1.9372
[Warmup] Epoch 75/100 - Loss: 1.7207
[Warmup] Epoch 76/100 - Loss: 1.7836
[Warmup] Epoch 77/100 - Loss: 1.8763
[Warmup] Epoch 78/100 - Loss: 1.8451
[Warmup] Epoch 79/100 - Loss: 1.6243
[Warmup] Epoch 80/100 - Loss: 1.5813
[Warmup] Epoch 81/100 - Loss: 1.5511
[Warmup] Epoch 82/100 - Loss: 1.4423
[Warmup] Epoch 83/100 - Loss: 1.2865
[Warmup] Epoch 84/100 - Loss: 1.5355
[Warmup] Epoch 85/100 - Loss: 1.4232
[Warmup] Epoch 86/100 - Loss: 1.2786
[Warmup] Epoch 87/100 - Loss: 1.2087
[Warmup] Epoch 88/100 - Loss: 1.5669
[Warmup] Epoch 89/100 - Loss: 1.4231
[Warmup] Epoch 90/100 - Loss: 1.1597
[Warmup] Epoch 91/100 - Loss: 1.1480
[Warmup] Epoch 92/100 - Loss: 1.1820
[Warmup] Epoch 93/100 - Loss: 1.1835
[Warmup] Epoch 94/100 - Loss: 1.3445
[Warmup] Epoch 95/100 - Loss: 1.1508
[Warmup] Epoch 96/100 - Loss: 1.2706
[Warmup] Epoch 97/100 - Loss: 1.0365
[Warmup] Epoch 98/100 - Loss: 1.1190
[Warmup] Epoch 99/100 - Loss: 1.1004
Checking heuristic stats
MD:  6 -> H: 6.6 (α=0.99)
MD:  4 -> H: 4.4 (α=0.99)
MD:  8 -> H: 8.8 (α=0.99)
MD: 14 -> H: 15.4 (α=0.99)

=== Curriculum Iteration 1/5 ===
Current α: 0.99, β: 0.0500
Found state at depth 30 with σₑ = 1.29 ≥ 0.5566166524310581
Found state at depth 30 with σₑ = 1.26 ≥ 0.3098220977635573
Found state at depth 30 with σₑ = 1.09 ≥ 0.1724521389063193
Found state at depth 30 with σₑ = 1.09 ≥ 0.0959897322626113
Found state at depth 30 with σₑ = 1.38 ≥ 0.05342948343976824
Found state at depth 30 with σₑ = 1.30 ≥ 0.029739740213364455
Found state at depth 30 with σₑ = 1.46 ≥ 0.016553634641732245
Found state at depth 30 with σₑ = 1.90 ≥ 0.009214028699847799
Found state at depth 30 with σₑ = 1.01 ≥ 0.005128681810312975
Found state at depth 30 with σₑ = 1.97 ≥ 0.002854709700640468
MD: 21 -> H: 23.1 (α=0.99) (Solved: 0/10)
MD: 21 -> H: 39.9 (α=0.99) (Solved: 1/10)

### Results Table:

Suboptimality Results for 15-puzzle:
α      Time       Subopt  Optimal
------------------------------------------
0.95   49.1  0.0%  30.0%



## **Efficiency Experiment**



### What it does:

Compares uncertainty-driven task generation (GTP) vs. fixed-step generation

Measures percentage of puzzles solved during training/testing


### Sample Output:

Found state at depth 30 with σₑ = 1.05 ≥ 1.0

Results Table:

Efficiency Results for 15-puzzle:
Method      Solved Train  Solved Test
------------------------------------
GTP         93.3%       60.6%
Inc 1       100%        38.6%


here is teh repo strcuture 


## Repository Structure

.
├── model.py               # Neural network implementations
├── sliding_puzzle.py      # 15-puzzle environment
├── practical_implementation.py  # Core training algorithms
├── suboptimality_experiment.py  # Table 1 reproduction
├── efficiency_experiment.py     # Table 2 reproduction
├── requirements.txt        # Python dependencies
└── README.md              # This document





## 📊 Results Interpretation
Key Findings
### Suboptimality

Achieves ~0% suboptimality but with computational overhead

Negative FFNN predictions indicate need for target normalization

Efficiency

GTP method solves 60.6% of test puzzles vs. 38.6% for fixed-step

Timeouts suggest hardware limitations or unoptimized search

Implementation Notes
✅ Successfully implemented:

Bayesian neural networks with epistemic/aleatoric uncertainty

Uncertainty-driven task generation

Curriculum learning framework

⚠️ Deviations from paper:

Python vs original C# implementation

🔧 Debugging Tips
For negative heuristics:

Normalize targets to [0,1] range

Add output activation (e.g., ReLU) in FFNN

*For IDA timeouts**:

Increase the time

Reduce max_depth parameter temporarily (easier puzzls)

Add progress logging in search algorithm

General checks:

Verify tensor shapes match between components (if adjustment where made to models)

Monitor loss curves during training