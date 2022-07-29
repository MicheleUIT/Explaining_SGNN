import numpy as np
# import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.ground_truth_loaders import load_mutag_gt, load_ba2_gt
from evaluation.AUCEvaluation import AUCEvaluation
from explainers.PGExplainer import PGExplainer
from explainers.MyExplainer import MyExplainer
from utils.plotting import plot, plot2, activation_hist

import sys
original_stdout = sys.stdout

def select_explainer(explainer, model, dataset, training_mask, training_loss, epochs, lr, reg_coefs, mask_thr, temp=None, sample_bias=None, gt_size=6, device='cuda'):
    if explainer == "PG":
        return PGExplainer(model, dataset, epochs=epochs, lr=lr, reg_coefs=reg_coefs, temp=temp, sample_bias=sample_bias, device=device)
    elif explainer == "MY":
        return MyExplainer(model, dataset, training_mask, training_loss, epochs=epochs, lr=lr, reg_coefs=reg_coefs, mask_thr=mask_thr, temp=temp, device=device)
    else:
        raise NotImplementedError("Unknown explainer type")


def run_qualitative_experiment(explainer, indices, labels, config, dataset):
    
    hist = []
    
    for idx in tqdm(indices):
        graph, expl, _ = explainer.explain(idx, labels[idx])
        plot(graph, expl, idx, dataset, config)

        hist.append(expl.numpy())
    
    plt.hist(np.concatenate(hist), bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.show()
            

def run_experiment(auc_eval, explainer, indices, labels):

    explanations = []
    scores = []
    for idx in tqdm(indices):
        graph, expl, score = explainer.explain(idx, labels[idx])
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


def run_experiment_BN(activations, config):
    layer_names = ['input', 'lin0', 'relu', 'lin1', 'output', 'soft_mask', 'gumbel_noise']

    for l in range(len(layer_names)):
        layer_activations = []
        for e in activations:
            s = e[l].size
            layer_activations.append(e[l].reshape(s))
        activation_hist(layer_activations, layer_names[l], config)


def explain(model, dataset, config, device='cuda'):
    
    model.eval()
    if dataset.name == 'mutag':
        explanation_labels = load_mutag_gt()
    else:
        explanation_labels = load_ba2_gt()
    labels = []
    indices = []
    i = 0
    for d in dataset:
        labels.append(d.y)
        if d.y == 0 and np.sum(explanation_labels[i]) > 0:
            indices.append(i)
        i+=1
    # Get explainer
    explainer = select_explainer(explainer=config.explainer,
                                 model=model,
                                 dataset=dataset,
                                 training_mask=config.training_mask,
                                 training_loss=config.training_loss,
                                 epochs=config.epochs,
                                 lr=config.lr,
                                 reg_coefs=[config.reg_size,
                                            config.reg_ent],
                                 mask_thr=config.mask_thr,
                                 temp=config.temp,
                                 sample_bias=config.sample_bias,
                                 gt_size=config.gt_size,
                                 device = device)
    
    loss = explainer.prepare(indices) # to use for PG explainer
    # loss, activations = explainer.prepare(indices) # to use for MY explainer
    # run_experiment_BN(activations, config)
    
    # run_qualitative_experiment(explainer, indices, labels, config, dataset)

    auc_evaluation = AUCEvaluation(explanation_labels, indices)
    auc, acc, fid, inf, num = run_experiment(auc_evaluation, explainer, indices, labels)
    masks_num = 0
    masks_dis = 0
    n_comp = 0
    
    # Experiments with subgraphs for ESAN
    # auc, acc, fid, inf, num, masks_num, masks_dis, n_comp = run_experiment_esan(auc_evaluation, explainer, indices, labels, config, dataset, config.mask_num, b_plot=False)
       
    return auc, acc, fid, inf, num, loss, masks_num, masks_dis, n_comp