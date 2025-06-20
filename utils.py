import datetime
import torch
import torch.nn as nn


def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def score_smoothing(scores, smoothing_window=3):
    smoothing_weights = torch.ones(1, 1, smoothing_window).to(scores.device) / smoothing_window
    smoothing_padding = smoothing_window // 2

    return nn.functional.conv1d(scores.unsqueeze(1), smoothing_weights, padding=smoothing_padding).squeeze(1)


def far_score(scores, labels, threshold=0.5):
    predicts = (scores > threshold).int()

    fp_count = ((predicts == 1) & (labels == 0)).sum().item()
    tn_count = ((predicts == 0) & (labels == 0)).sum().item()

    return fp_count / (fp_count + tn_count)


def iou_score(sequence1, sequence2, eps=1e-6):
    intersection_score = (sequence1 * sequence2).sum()

    range1_score = sequence1.sum()
    range2_score = sequence2.sum()

    return (intersection_score + eps) / (range1_score + range2_score - intersection_score + eps)


def dice_score(sequence1, sequence2, eps=1e-6):
    intersection_score = (sequence1 * sequence2).sum()

    range1_score = sequence1.sum()
    range2_score = sequence2.sum()

    return (2 * intersection_score + eps) / (range1_score + range2_score + eps)


def bidirectional_dice_score(sequence1, sequence2, alpha, eps=1e-6):
    return alpha * dice_score(sequence1, sequence2, eps) + (1 - alpha) * dice_score(1 - sequence1, 1 - sequence2, eps)
