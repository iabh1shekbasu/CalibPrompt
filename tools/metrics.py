'''
    https://github.com/MiaoXiong2320/ProximityBias-Calibration/blob/main/utils/metrics.py
    https://github.com/markus93/NN_calibration/blob/eb235cdba006882d74a87114a3563a9efca691b7/scripts/utility/evaluation.py
    https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
    
    This file contains the code for evaluation metrics:
    - ECE 
    - MCE
    - Dist-aware ECE
    - Adaptive ECE
    ...
'''

import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time, pdb
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import average_precision_score, roc_auc_score, auc
import sys
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin



   
# def ECE(conf, pred, gt, conf_bin_num = 10):

#     """
#     Expected Calibration Error
    
#     Args:
#         conf (numpy.ndarray): list of confidences
#         pred (numpy.ndarray): list of predictions
#         true (numpy.ndarray): list of true labels
#         bin_size: (float): size of one bin (0,1)  
        
#     Returns:
#         ece: expected calibration error
#     """
#     df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
#     df['correct'] = (df.pred == df.ys).astype('int')
    

#     bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
#     df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
#     # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
#     # groupy by knn + conf
#     group_acc = df.groupby(['conf_bin'])['correct'].mean()
#     group_confs = df.groupby(['conf_bin'])['conf'].mean()
#     counts = df.groupby(['conf_bin'])['conf'].count()
#     ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
#     return ece

def ECE(conf, pred, gt, conf_bin_num = 10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    bins = np.linspace(0, 1, conf_bin_num+1)
    bin_indices = np.digitize(conf, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(conf_bin_num):

        in_bin = bin_indices == i

        if np.sum(in_bin) > 0:
            accuracy = np.mean(gt[in_bin] == pred[in_bin])
            mean_confidence = np.mean(conf[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)


    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)


    weights = np.histogram(conf, bins)[0] / len(conf)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))
        
    return ece
     
def PIECE(conf, knndist, pred, gt, dist_bin_num =10, conf_bin_num = 10, knn_strategy='quantile'):

    """
    Proximity-Informed Expected Calibration Error 
    
    Args:
        conf (numpy.ndarray): list of confidences
        knndist (numpy.ndarray): list of distances of which a sample to its K nearest neighbors
        pred (numpy.ndarray): list of predictions
        gt (numpy.ndarray): list of true labels
        dist_bin_num: (float): the number of bins for knndist
        conf_bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    
    
    df = pd.DataFrame({'ys':gt, 'knndist':knndist, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['knn_bin'] = KBinsDiscretizer(n_bins=dist_bin_num, encode='ordinal',strategy=knn_strategy).fit_transform(knndist[:, np.newaxis])
    
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['knn_bin', 'conf_bin'])['correct'].mean()
    group_confs = df.groupby(['knn_bin', 'conf_bin'])['conf'].mean()
    counts = df.groupby(['knn_bin', 'conf_bin'])['conf'].count()
    ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    # group by only knn
    # group_acc = df.groupby(['knn_bin'])['correct'].mean()
    # group_confs = df.groupby(['knn_bin'])['conf'].mean()
    # counts = df.groupby(['knn_bin'])['conf'].count()
    # ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    
    # n = len(conf)
    # ece = 0  # Starting error
    # upper_bounds = np.arange(conf_bin_size, 1+conf_bin_size, conf_bin_size)  # Get bounds of bins
    # for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
    #     acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-conf_bin_size, conf_thresh, conf, pred, gt)        
    #     ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece


def MCE(conf, pred, gt, conf_bin_num = 10):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        mce: maximum calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')

    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    mce = (np.abs(group_acc - group_confs) * counts / len(df)).max()
        
    return mce



def AdaptiveECE(conf, pred, gt, conf_bin_num=10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ace: expected calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='quantile').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    ace = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
    return ace

def ECE_KDE(conf, pred, gt, p=1, bandwidth=None):
    """
    Expected Calibration Error using Kernel Density Estimation (ECE-KDE)
    
    This implements a simplified version of the ECE-KDE metric from the paper:
    "A Consistent and Differentiable Lp Canonical Calibration Error Estimator" (NeurIPS 2022)
    
    The implementation uses a Gaussian kernel for simplicity and computational efficiency,
    rather than the Dirichlet/Beta kernels in the original paper. This simplification is
    appropriate for medical imaging datasets with few-shot learning for several reasons:
    
    1. Computational Efficiency: The Gaussian kernel is faster to compute and has lower 
       memory requirements than the Dirichlet kernel, important for iterative training.
    
    2. Top-Label Focus: For medical diagnosis tasks, top-label (confidence) calibration 
       is often the primary concern, making the full canonical calibration unnecessary.
    
    3. Few-Shot Robustness: In few-shot learning (e.g., 8-shot), simpler models with fewer
       parameters tend to be more robust, and the same applies to calibration metrics.
    
    4. Adaptive Bandwidth: The implementation uses an adaptive bandwidth based on dataset 
       size, which is particularly important for few-shot learning where test sets can vary.
    
    The key difference from traditional binning-based ECE is that KDE provides a smooth, 
    continuous estimate of the relationship between confidence and accuracy, avoiding
    artifacts from arbitrary bin boundaries.
    
    Args:
        conf (numpy.ndarray): list of confidences (max probability values)
        pred (numpy.ndarray): list of predictions (class indices)
        gt (numpy.ndarray): list of true labels (class indices)
        p (int): order of the norm (1 for L1 norm, 2 for L2 norm)
        bandwidth (float): optional manual bandwidth parameter, auto-selected if None
        
    Returns:
        ece_kde: expected calibration error using KDE
    """
    
    # Convert inputs to torch tensors if needed
    if not isinstance(conf, torch.Tensor):
        conf = torch.tensor(conf, dtype=torch.float32)
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.long)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt, dtype=torch.long)
    
    # Calculate accuracy (1 if correct, 0 if wrong)
    acc = (pred == gt).float()
    
    # Get number of samples
    n = len(conf)
    
    # Set bandwidth based on dataset size (determined from test set)
    # For few-shot learning scenarios, larger bandwidths prevent overfitting
    if bandwidth is None:
        if n < 100:  # Very small datasets like test sets for few-shot learning
            bandwidth = 0.3
        elif n < 500:  # Small test sets
            bandwidth = 0.2
        elif n < 2000:  # Medium test sets
            bandwidth = 0.1
        else:  # Large test sets
            bandwidth = 0.05
    
    # Calculate kernel matrix using Gaussian kernel
    # This is simpler than the Beta/Dirichlet kernels in the paper but still effective
    conf_expanded = conf.unsqueeze(1)
    diff = conf_expanded - conf_expanded.T
    kernel = torch.exp(-(diff**2) / (2 * bandwidth**2))
    kernel.fill_diagonal_(0)  # exclude self-comparisons for leave-one-out estimation
    
    # Normalize kernel to ensure proper weighting
    kernel_sum = kernel.sum(dim=1, keepdim=True)
    kernel_sum = torch.clamp(kernel_sum, min=1e-10)  # avoid division by zero
    kernel_norm = kernel / kernel_sum
    
    # Estimate accuracy for each confidence value using KDE
    estimated_acc = torch.matmul(kernel_norm, acc)
    
    # Calculate ECE-KDE using Lp norm (default p=1 for L1 norm)
    ece_kde = torch.mean(torch.abs(conf - estimated_acc)**p)
    
    return ece_kde.item()