import torch


def format_segmentation_masks(y_true, y_pred, threshold):
    """Bring the ground truth and the prediction masks to the same format (C, W, H) with values 1.0 or 0.0"""
    pred_onehot = torch.where(y_pred > threshold, 1.0, 0.0)
    y_gt_i = y_true.clone().unsqueeze(0).type(torch.int64)
    gt_onehot = torch.zeros_like(pred_onehot)
    gt_onehot.scatter_(0, y_gt_i, 1.0)
    return gt_onehot, pred_onehot


def segmentation_counts_per_class(y_true_onehot, y_pred_onehot):
    """Compute the ground truth, predicted and intersection areas per class for segmentation metrics"""
    tp_onehot = torch.logical_and(y_true_onehot, y_pred_onehot)
    tp_count_per_class = torch.sum(tp_onehot, dim=[1, 2])
    gt_count_per_class = torch.sum(y_true_onehot, dim=[1, 2])
    pred_count_per_class = torch.sum(y_pred_onehot, dim=[1, 2])
    return tp_count_per_class, gt_count_per_class, pred_count_per_class


def segmentation_counts_micro(y_true_onehot, y_pred_onehot):
    """Compute the micro averaged ground truth, predicted and intersection areas for segmentation metrics"""
    tp_onehot = y_true_onehot * y_pred_onehot
    tp_count_per_class = torch.sum(tp_onehot)
    gt_count_per_class = torch.sum(y_true_onehot)
    pred_count_per_class = torch.sum(y_pred_onehot)
    return tp_count_per_class, gt_count_per_class, pred_count_per_class
