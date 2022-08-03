import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

# from datasets.ground_truth_loaders import load_mutag_gt, load_ba2_gt
from evaluation.AUCEvaluation import AUCEvaluation
# from explainers.PGExplainer import PGExplainer
from explainer import MyExplainer
from utils.plotting import plot, plot2, activation_hist
from data import MutagGTDataset, policy2transform
from torch_geometric.data import DataLoader

import sys
original_stdout = sys.stdout



def run_experiment(auc_eval, explainer, test_loader):

    explanations = []
    scores = []
    for data in tqdm(test_loader):
        graph, expl, score = explainer.explain(data)
        explanations.append((graph, expl))
        scores.append(np.asarray(score))

    auc_score = auc_eval.get_score(explanations)

    scores = np.asarray(scores)
    acc = scores[:,0].mean()
    fid = scores[:,1].mean()
    inf = scores[:,2].mean()
    n =   scores[:,3].mean()
    
    return auc_score, acc, fid, inf, n


def run_experiment_esan(auc_eval, explainer, indices, labels, config, dataset, mask_num=10, b_plot=False):
    
    explanations = []
    scores = []
    for idx in tqdm(indices):
        graph, expl, score, masks, mask_acc = explainer.explain2(idx, labels[idx], config.subgraph_policy, mask_num, 
                                                                 temperature=config.subgraph_temp, 
                                                                 threshold=config.subgraph_threshold, 
                                                                 delta=config.subgraph_delta)
        explanations.append((graph, expl))
        scores.append(np.asarray(score))

        if b_plot:
            # Plot all subgraphs
            i = 0
            for mask in masks:
                plot(graph, mask, idx, dataset, config, multiple=True, mask_idx=i)
                i += 1
                        
            # Plot graph with a colorscale for the frequency
            plot2(graph, mask_acc, idx, config, show=False)

    auc_score = auc_eval.get_score(explanations)

    scores = np.asarray(scores)
    acc = scores[:,0].mean()
    fid = scores[:,1].mean()
    inf = scores[:,2].mean()
    n =   scores[:,3].mean()
    masks_num = scores[:,4].mean()
    masks_dis = scores[:,5].mean()
    n_comp = scores[:,6].mean()

    # with open('qualitative/logs.txt', 'a') as f:
    #     sys.stdout = f
    #     print("\nmasks_num: ", scores[:,4], "\nmasks_dis: ", scores[:,5])
    #     sys.stdout = original_stdout

    return auc_score, acc, fid, inf, n, masks_num, masks_dis, n_comp


def filter_gt(data):
    return True if torch.sum(data.edge_gt) > 0 else False


def explain(model, args, config, device='cuda'):
    
    dataset = MutagGTDataset(root="dataset/prefiltered/" + args.policy,
                            name=args.dataset,
                            pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops, dataset_name=args.dataset, device=device),
                            pre_filter=filter_gt
                            )

    orig_dataset = MutagGTDataset(root="dataset/prefiltered/" + "original",
                            name=args.dataset,
                            pre_transform=None,
                            pre_filter=filter_gt
                            )
    
    train_loader = DataLoader(dataset, args.batch_size, shuffle=True)
    test_subgraph_loader = DataLoader(dataset, batch_size=1, shuffle=False) # batch?
    test_original_loader = DataLoader(orig_dataset, batch_size=1, shuffle=False) # batch?

    # model.eval()
    # if dataset.name == 'mutag':
    #     explanation_labels = load_mutag_gt()
    # else:
    #     explanation_labels = load_ba2_gt()
    # labels = []
    # indices = []
    # i = 0
    # for d in dataset:
    #     labels.append(d.y)
    #     if d.y == 0 and np.sum(explanation_labels[i]) > 0:
    #         indices.append(i)
    #     i+=1
    # Get explainer
    explainer = MyExplainer(config.training_mask, config.training_loss, epochs=config.epochs, lr=config.lr, 
                            reg_coefs=config.reg_coefs, mask_thr=config.mask_thr, temp=config.temp, device=device)
    
    explainer.prepare(model)
    explainer.train(train_loader)

    auc_evaluation = AUCEvaluation(explanation_labels, indices)

    auc, acc, fid, inf, num = run_experiment(auc_evaluation, explainer, test_subgraph_loader)
          
    return auc, acc, fid, inf, num