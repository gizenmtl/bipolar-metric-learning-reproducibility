# Bipolar Metric Learning Reproducibility Package

This repository contains the manuscript-specific reproducibility code for:

**Bipolar Metric Spaces for Geometric Regularization in Distance-Based Learning**

The repository provides research code for reproducing the BP-KNN experiments, baseline KNN comparisons, pruning-efficiency analysis, decision-stability analysis, and memory-scalability figures reported in the manuscript.

## Scope of this repository

This repository is intended as a reproducibility package for the manuscript version submitted to *Mathematics*. It is not intended as a production-ready software package.

The repository includes:

- BP-KNN implementation with a bipolar sign-space structural filter
- Standard KNN baseline comparison scripts
- Euclidean, Manhattan, Chebyshev, and Minkowski distance options
- Accuracy, F1-score, ROC-AUC, runtime, and memory evaluation utilities
- Figure 4 memory scalability script
- Figure 5 decision stability script
- Figure 6 pruning efficiency script
- Dataset loading and preprocessing utilities
- Environment requirements and reproduction instructions

Future extensions, optimized variants, and ongoing development versions are maintained separately and are outside the scope of this reproducibility release.

## Repository structure

```text
src/
  bpknn.py
  data_utils.py
  evaluation.py
  plotting.py

experiments/
  run_bpknn_baseline_comparison.py
  figure4_memory_scalability.py
  figure5_decision_stability.py
  figure6_pruning_efficiency.py

data/
  README.md

outputs/
  .gitkeep

requirements.txt
README.md
CITATION.cff