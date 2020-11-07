# %%
import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef


def compute_reg_metrics(preds, targets):
  abs_err = np.abs(preds - targets)
  cae = abs_err.sum()
  mae = cae / len(abs_err)
  mse = (abs_err ** 2).sum() / len(abs_err)
  return cae, mae, mse

def compute_cls_metrics(preds, targets):
  qk = cohen_kappa_score(preds, targets, weights="quadratic")
  mcc = matthews_corrcoef(preds, targets)
  return qk, mcc