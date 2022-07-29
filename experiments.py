# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:00:01 2022

@author: mgphy
"""

import torch
import numpy as np
import wandb
import pandas as pd

from utils import get_model, get_data
from train_esan import load_best_model

# from torch_geometric.loader import DataLoader
# from datasets.dataset_loaders import load_dataset
# from tasks.training import load_best_model
# from tasks.replication import explain


#%%
# WANDB configurations

wandb.init(project="expl_esan",
            entity="tromso",
            config = {
                "dataset": "mutag", # mutag
                "model": "GCN", # GCN
                "explainer": "PG",
                "training_mask": "soft",
                "training_loss": "hard", # entropy term in s_loss doesn't make sense
                "sample_bias": 0.0,
                "seed": 10,
                "epochs": 50,
                "lr": 0.005, #0.005, # 0.001
                "temp": [5.0, 1.0, 10.0],# [5.0, 1.0, 5.0],
                "reg_size": 0.05, #0.05, # 0.1
                "reg_ent": 1,
                "gt_size": 10,
                "mask_thr": 0.5,
                "mask_num": 10,
                "subgraph_policy": 'progr', # noise, threshold, progr
                "subgraph_temp": 1.0,
                "subgraph_threshold": 0.5,
                "subgraph_delta": 0.3
                }
            )
config = wandb.config


#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = config.dataset
model_name = config.model
dataset = load_dataset(dataset_name)
input_dim = max(dataset.n_features, 1)
h_dim = 32
aucs = []
accs = []
fids = []
infs = []
sizes = []
losses = []
masks_num_l = []
masks_dis_l = []
components = []


train_loader, _, valid_loader, _, attributes = get_data(config, fold_idx, device)
in_dim, out_dim, task_type, eval_metric = attributes

model = get_model(config, in_dim, out_dim, device)
model = load_best_model(dataset.name, model, "gcn_"+dataset_name+f"_{j}", device=device)

model.to(device)  

# Change seed for explainer only
for s in config.seed:
    torch.manual_seed(s)
    np.random.seed(s)
      
    auc, acc, fid, inf, n, loss, masks_num, masks_dis, n_comp = explain(model, dataset, config, device)
    # auc, fid, inf, n =  explain(model, dataset, exp_config.args.explainer, device)
    wandb.log({"AUC": auc, "accuracy": acc, "fidelity": fid, "infidelity": inf, "hard_mask": n, "loss": loss})
    aucs.append(auc)
    accs.append(acc)
    fids.append(fid)
    infs.append(inf)
    sizes.append(n)
    losses.append(loss[0])
    masks_num_l.append(masks_num)
    masks_dis_l.append(masks_dis)
    components.append(n_comp)

s_auc = np.asarray(aucs).std()
m_auc = np.asarray(aucs).mean()
s_acc = np.asarray(accs).std()
m_acc = np.asarray(accs).mean()
s_fid = np.asarray(fids).std()
m_fid = np.asarray(fids).mean()
s_inf = np.asarray(infs).std()
m_inf = np.asarray(infs).mean()
s_size = np.asarray(sizes).std()
m_size = np.asarray(sizes).mean()
s_loss = np.asarray(losses).std()
m_loss = np.asarray(losses).mean()
s_masks = np.asarray(masks_num_l).std()
m_masks = np.asarray(masks_num_l).mean()
s_masks_dis = np.asarray(masks_dis_l).std()
m_masks_dis = np.asarray(masks_dis_l).mean()
s_comp = np.asarray(components).std()
m_comp = np.asarray(components).mean()


# wandb.log({"AUC_mean": m_auc, "AUC_std": s_auc, "fidelity_mean": m_fid, "fidelity_std": s_fid, 
#             "infidelity_mean": m_inf, "infidelity_std": s_inf, "size_mean": m_size, "size_std": s_size})
            # "loss_mean": m_loss, "loss_std": s_loss}, commit=True)
#%%
df = pd.DataFrame({"AUC":aucs, "Accuracy":accs, "Fidelity":fids, "Infidelity":infs, "Size":sizes, "Loss":losses, "mask_num":masks_num_l, "mask_diff":masks_dis_l, "components": components})
with pd.ExcelWriter(f"qualitative/PG_results.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name=f"{config.model}_{dataset_name}_{config.subgraph_policy}", index=False) 

print(f"AUC: {m_auc} \pm {s_auc}")
print(f"Accuracy: {m_acc} \pm {s_acc}")
print(f"FID: {m_fid} \pm {s_fid}")
print(f"INF: {m_inf} \pm {s_inf}")
print(f"Mask size: {m_size} \pm {s_size}")
print(f"Loss: {m_loss} \pm {s_loss}")
print(f"Mask number: {m_masks} \pm {s_masks}")
print(f"Mask differences: {m_masks_dis} \pm {s_masks_dis}")
print(f"Components: {m_comp} \pm {s_comp}")