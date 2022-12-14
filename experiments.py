# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:00:01 2022

@author: mgphy
"""
#%%

import torch
import numpy as np
import wandb
import pandas as pd
import random

from esan_utils.utils import get_model
from train_esan import load_best_model

from explainer_utils.replication import explain
from esan_utils.data import MutagGTDataset, BA2GTDataset, policy2transform, filter_gt


#%%
# WANDB configurations

config_expl = {
                "training_mask": "hard",
                "expl_seed": 1,
                "expl_epochs": 35,
                "lr": 0.0002, 
                "temp0": 2.0,
                "temp1": 5.0,
                "temp2": 2.0,
                "size_reg": 0.1, 
                "mask_thr": 0.1,
                }

config_esan = {
                'gnn_type': 'pgegin',
                'num_layer': 4,
                'emb_dim': 32,
                'batch_size': 32,
                'dataset': 'Mutagenicity',
                'jk': 'concat',
                'drop_ratio': 0.,
                'channels': '32-32',
                'policy': 'node_deleted',
                'num_hops': 2,
                'model': 'deepsets',
                'seed': 0
                }

config = config_expl
config.update(config_esan)

wandb.init(project="expl_esan",
            entity="tromso",
            config=config
            )
config = wandb.config


#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config.dataset == "Mutagenicity":
    dataset = MutagGTDataset(root="dataset/prefiltered/" + config.policy,
                                name=config.dataset,
                                pre_transform=policy2transform(policy=config.policy, num_hops=config.num_hops, dataset_name=config.dataset, device=device),
                                pre_filter=filter_gt
                                )
elif config.dataset == "ba2":
    dataset = BA2GTDataset(root="dataset/prefiltered/" + config.policy,
                            name=config.dataset,
                            pre_transform=policy2transform(policy=config.policy, num_hops=config.num_hops, dataset_name=config.dataset, device=device),
                            pre_filter=filter_gt
                            )

aucs = []
accs = []
fids = []
infs = []
sizes = []

in_dim = dataset.num_features
out_dim = dataset.num_tasks

model = get_model(config, in_dim, out_dim, device)
model = load_best_model(config, model, device=device)

model.to(device)

# plot masks?
b_plot = False
# print results in excel?
b_results = False

# Change seed for explainer only
for s in range(config.expl_seed):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
      
    auc, acc, fid, inf, n = explain(model, dataset, config, s, b_plot, device)

    wandb.log({"AUC": auc, "accuracy": acc, "fidelity": fid, "infidelity": inf, "size": n})
    aucs.append(auc)
    accs.append(acc)
    fids.append(fid)
    infs.append(inf)
    sizes.append(n)

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

wandb.log({"AUC": m_auc})
wandb.log({"AUC_mean": m_auc, "AUC_std": s_auc, "fidelity_mean": m_fid, "fidelity_std": s_fid, 
            "infidelity_mean": m_inf, "infidelity_std": s_inf, "size_mean": m_size, "size_std": s_size})
#%%
if b_results:
    df = pd.DataFrame({"AUC":aucs, "Accuracy":accs, "Fidelity":fids, "Infidelity":infs, "Size":sizes})
    with pd.ExcelWriter(f"results/results_{config.dataset[:3]}_{config.model[:3]}.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=f"{config.gnn_type[:3]}_{config.policy}", index=False) 

print(f"AUC: {m_auc} \pm {s_auc}")
print(f"Accuracy: {m_acc} \pm {s_acc}")
print(f"FID: {m_fid} \pm {s_fid}")
print(f"INF: {m_inf} \pm {s_inf}")
print(f"Mask size: {m_size} \pm {s_size}")