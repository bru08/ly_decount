# %%
import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import bisect

def compute_reg_metrics(res_dict):
  preds = np.array(res_dict["pred"])
  targets = np.array(res_dict["trgt"])
  abs_err = np.abs(preds - targets)
  cae = abs_err.sum()
  mae = cae / len(abs_err)
  mse = (abs_err ** 2).sum() / len(abs_err)
  return cae, mae, mse

def grade_opbg(score, breakpoints=[0,5,10,20,50,200], grades=list(range(7))):
  """
  Convert number of lymphocytes into one of the seven classes for the opbg
  """
  i = bisect.bisect_left(breakpoints, score)
  return grades[i]

def compute_cls_metrics(res_dict):
  preds = np.array([grade_opbg(elem) for elem in res_dict["pred"]])
  targets = np.array([grade_opbg(elem) for elem in res_dict["trgt"]])

  qk = cohen_kappa_score(preds, targets, weights="quadratic")
  mcc = matthews_corrcoef(preds, targets)
  return qk, mcc

