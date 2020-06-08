import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import pearsonr, spearmanr
from statistics import *
from scipy.special import softmax

def metric_multi_accuracy(logits, labels, options_num):
  logits = np.reshape(softmax(logits, -1)[:,1], (len(logits)//options_num, options_num))
  labels = np.argmax(np.reshape(labels, (len(labels)//options_num, options_num)),-1)
  return metric_accuracy(logits, labels)

def metric_accuracy(logits, labels):
  predicts = np.argmax(logits, axis=1)
  return accuracy_score(labels, predicts)

def metric_f1(logits, labels):
  predicts = np.argmax(logits, axis=1)
  return f1_score(labels, predicts)

def metric_macro_f1(logits, ground_truth, labels=[0,1]):
  predicts = np.argmax(logits, axis=1)
  f1=[]
  for l in labels:
    binary_g = (ground_truth==l).astype(np.int)
    binary_p = (predicts==l).astype(np.int)
    f1.append(f1_score(binary_g, binary_p))
  return float(np.mean(f1))

def metric_mcc(logits, labels):
  predicts = np.argmax(logits, axis=1)
  return matthews_corrcoef(labels, predicts)
