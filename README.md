# FYS-STK 3155/4155

Coursework for FYS-STK 3155/4155 at the University of Oslo, autumn 2026.

## Project 1 — Regression, resampling and gradient descent

Polynomial regression on Runge's function f(x) = 1/(1+25x²) with OLS, Ridge and
Lasso; bias–variance analysis with the bootstrap; model selection with k-fold
cross-validation; and gradient-based optimisation (plain GD, momentum, AdaGrad,
RMSProp, Adam and stochastic gradient descent).


Submitted by **Gard Kvalsvik Lilleås**. The project was started as a group
project with **Parisa Amin**, who wrote parts b.5, c, e and g of the notebook.
Parts d, f, h and i were written by Gard; parts a and b and `regression.py` were
developed jointly.

```
Project1/
  code/      regression.py and the notebooks
  results/   the figures used in the report
  report/    bibliography (the report itself is written in Overleaf)
```

The Jupyter notebook with the code for this project is `Project1_gard.ipynb`. Library functions are found in `regression.py`.

## Running the code

Run all cells from the top ("Restart and Run all").

```
pip install -r requirements.txt
```

All random seeds are fixed, so the figures can be reproduced from the notebooks.
