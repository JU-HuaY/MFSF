import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, confusion_matrix, matthews_corrcoef
import numpy as np
import torch


def caculate_metric(pred_bs, labels):
    # pred_bs = pred_bs.to('cpu').data.numpy()
    # labels = labels.to('cpu').data.numpy()
    pre_y = list(map(lambda x: np.argmax(x), pred_bs))
    pre_prob = list(map(lambda x: x[1], pred_bs))

    tn, fp, fn, tp = confusion_matrix(labels, pre_y).ravel()
    Specificity = tn / (tn + fp)

    MCC = matthews_corrcoef(labels, pre_y)
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1_score = (2 * Precision * Recall) / (Precision + Recall)

    fpr, tpr, thresholds = roc_curve(labels, pre_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)

    return Specificity, Recall, Precision, F1_score, MCC, AUC

# if __name__ == "__main__":
#     pre = torch.tensor([[0, 1], [0.3, 0.7], [0.25, 0.8], [0.6, 0.4], [0.7, 0.3]])
#     label = torch.tensor([1, 1, 0, 1, 0])
#     print(caculate_metric(pre, label))
