import numpy as np
from sklearn.metrics import roc_auc_score


def evaluation_auc(explanations, ground_truths):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []

    for n in range(len(ground_truths)): # Use idx for explanation list and indices for ground truth list

        # Select explanation
        mask = explanations[n][1].detach().numpy()
        graph = explanations[n][0].detach().numpy()
        edge_labels = ground_truths[n]
        
        for edge_idx in range(len(edge_labels)):
            edge_ = graph.T[edge_idx]
            if edge_[0] == edge_[1]:
                continue
            t = np.where((graph.T == edge_.T).all(axis=1))

            predictions.append(mask[t][0])
            ground_truth.append(edge_labels[edge_idx])

    score = roc_auc_score(ground_truth, predictions)
    return score
