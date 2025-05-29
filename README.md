# Reproducibility Study: Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics

This repository contains a Python implementation reproducing the paper:  
**"Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics"** by Ofir Maron and Benjamin Rosman (2020).  

[Original Paper](https://www.raillab.org/publication/marom-2020-utilising/) | [Original C# Code](https://github.com/OfirMaron/LearnHeuristicWithUncertainty)

---

## Table of Contents
1. [Installation](#installation)
2. [Running Experiments](#running-experiments)
3. [Repository Structure](#repository-structure)
4. [Experimental Setup](#experimental-setup)
5. [Interpretation](#interpretation)
6. [Key Implementation Details](#key-implementation-details)
7. [Troubleshooting & Debugging Tips](#troubleshooting--debugging-tips)

---

## Installation

### Prerequisites
- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended) or `pip`
- PyTorch
- NumPy
- SciPy

### Setup
```bash
git clone https://github.com/MichaelFalk1/AI-Assignment-Reproducibility-.git # <-- Add your repo link here
cd reproducibility-assignment

# With Poetry (recommended):
poetry install

# With pip:
pip install -r requirements.txt
```

---

## Running Experiments

### 1. Suboptimality Experiment

Trains the Weighted Uncertainty Neural Network (WUNN) and Feedforward Neural Network (FFNN), then evaluates heuristic quality on 15-puzzle benchmarks.

```bash
python src/suboptimality_experiment.py
```

**Outputs:**
- Logs for WUNN warmup and curriculum training
- FFNN training logs
- Suboptimality results table for various $\alpha$ values

### 2. Efficiency Experiment

Compares uncertainty-driven task generation (GTP) vs. fixed-step generation. Measures percentage of puzzles solved during training/testing.

```bash
python src/efficiency_experiment.py
```

**Outputs:**
- Efficiency results table for GTP and fixed-step methods

---

## Repository Structure

```
.
├── model.py                     # Neural network implementations (WUNN, FFNN)
├── sliding_puzzle.py            # 15-puzzle environment and utilities
├── practical_implementation.py  # Core training algorithms and curriculum learning
├── suboptimality_experiment.py  # Suboptimality experiment (Table 1 reproduction)
├── efficiency_experiment.py     # Efficiency experiment (Table 2 reproduction)
└── README.md                    # This document
```

---

## Experimental Setup

All experiments were conducted using a created Python codebase, which implements the 15-puzzle environment, uncertainty-aware heuristic learning, and evaluation scripts. The experiments were run on a standard CPU machine using Python 3.11, PyTorch, NumPy, and SciPy.

- **Puzzle Environment:** The 15-puzzle is implemented in `sliding_puzzle.py`, supporting random shuffling, legal moves, and state encoding.
- **Model Training:** The Weighted Uncertainty Neural Network (WUNN) is trained using curriculum learning, where tasks are generated adaptively based on epistemic uncertainty. The final heuristic is computed using a small feedforward neural network (FFNN) that takes the WUNN's mean and uncertainty as input.
- **Experiments:**
  - **Efficiency Experiment:** Evaluates the percentage of solved puzzles on both training and test sets, comparing the learned heuristic to fixed-step baselines.
  - **Suboptimality Experiment:** Evaluates the average suboptimality (percentage above optimal cost), percentage of optimal solutions, average search time, and average number of nodes generated, for various uncertainty thresholds ($\alpha$).
- **Evaluation Metrics:**
  - **Solved Percentage:** The fraction of puzzles solved within the time limit.
  - **Suboptimality:** The average percentage by which the solution cost exceeds the optimal cost.
  - **Optimality:** The percentage of solutions that are optimal.
  - **Average Time:** The mean time taken to solve each puzzle.
  - **Nodes Generated:** The average number of nodes expanded during search.
- **Reproducibility:** All random seeds are set at the start of each experiment for reproducibility. The code is modular and can be run directly using the provided scripts.

---

## Interpretation

### Suboptimality

- Achieves near 0% suboptimality for high $\alpha$ values, with some computational overhead.
- Negative FFNN predictions may occur if targets are not normalized; normalization is now included.

**Sample Output:**
```
Suboptimality Results for 15-puzzle:
α      Time   Nodes   Subopt  Optimal
------------------------------------------
0.95   49.1   1234    0.0%    30.0%
...
```

### Efficiency

- GTP method solves a higher percentage of test puzzles compared to fixed-step baselines.

**Sample Output:**
```
Efficiency Results for 15-puzzle:
Method      Solved Train  Solved Test
------------------------------------
GTP         93.3%         60.6%
Inc 1       100%          38.6%
...
```

---

## Key Implementation Details

- **Uncertainty-Aware Heuristics:**  
  Uses a Weighted Uncertainty Neural Network (WUNN) to estimate both mean and uncertainty for heuristic values.

- **Curriculum Learning:**  
  Tasks are generated adaptively based on epistemic uncertainty to focus learning on challenging states.

- **Node Counting:**  
  All search algorithms return the number of nodes generated for fair comparison.

- **Evaluation Metrics:**  
  - Solved Percentage
  - Suboptimality (percentage above optimal cost)
  - Optimality (percentage of optimal solutions)
  - Average Time per puzzle
  - Average Nodes Generated

- **Reproducibility:**  
  All random seeds are set at the start of each experiment for reproducibility. The code is modular and can be run directly using the provided scripts.

---

## Troubleshooting & Debugging Tips

- **Negative Heuristics:**  
  - Normalize targets to [0,1] range (now handled in FFNNTrainer).
  - Add output activation (e.g., ReLU or sigmoid) in FFNN.

- **IDA* Timeouts:**  
  - Increase the time limit (`t_max`).
  - Reduce `max_depth` parameter temporarily (for easier puzzles).
  - Add progress logging in the search algorithm.

- **General Checks:**  
  - Verify tensor shapes match between components (especially after model changes).
  - Monitor loss curves during training for signs of divergence or overfitting.