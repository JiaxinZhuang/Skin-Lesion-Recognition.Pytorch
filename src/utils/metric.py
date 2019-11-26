"""Metric.

    Mean class recall.
"""

import numpy as np


def mean_class_recall(y_true, y_pred):
    """Mean class recall.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    class_recall = []
    target_uniq = np.unique(y_true)

    for label in target_uniq:
        indexes = np.nonzero(label == y_true)[0]
        recall = np.sum(y_true[indexes] == y_pred[indexes]) / len(indexes)
        class_recall.append(recall)
    return np.mean(class_recall)


if __name__ == "__main__":
    y_pred = [0, 0, 1, 1, 2, 2, 2]
    y_true = [0, 0, 0, 0, 1, 2, 2]
    print(mean_class_recall(y_true, y_pred))
