import numpy as np
import matplotlib.pyplot as plt
import torch

from explainer_utils.explainer import MyExplainer
from data import MutagGTDataset, BA2GTDataset, filter_gt
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from explainer_utils.plotting import plot
from tqdm import tqdm



def run_experiment(explainer, test_subgraph_loader, test_original_loader, config, b_plot):
    
    accs, fids, infs, sums, auc, explanations = explainer.explain(test_subgraph_loader, test_original_loader)

    acc = np.asarray(accs).mean()
    fid = np.asarray(fids).mean()
    inf = np.asarray(infs).mean()
    n   = np.asarray(sums).mean()

    if b_plot:
        for idx, (graph, _, mask) in tqdm(enumerate(explanations)):
            plot(graph, mask, idx, config, False)
     
    return acc, fid, inf, n, auc


def explain(model, dataset, config, s, b_plot = False, device='cuda'):
    
    if config.dataset == "Mutagenicity":
        orig_dataset = MutagGTDataset(root="dataset/prefiltered/" + "original",
                                        name=config.dataset,
                                        pre_transform=None,
                                        pre_filter=filter_gt
                                        )
    elif config.dataset == "ba2":
        orig_dataset = BA2GTDataset(root="dataset/prefiltered/" + "original",
                                        name=config.dataset,
                                        pre_transform=None,
                                        pre_filter=filter_gt
                                        )
    
    N = len(dataset)
    g = torch.Generator()
    g.manual_seed(s)
    train_dataset, test_dataset = random_split(dataset,[N*70//100,N*30//100],generator=g)
    g.manual_seed(s)
    _, test_original_dataset = random_split(orig_dataset,[N*70//100,N*30//100],generator=g)
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, follow_batch=['subgraph_idx', 'original_x'])
    test_subgraph_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, follow_batch=['subgraph_idx', 'original_x']) 
    test_original_loader = DataLoader(test_original_dataset, batch_size=1, shuffle=False, follow_batch=['subgraph_idx', 'original_x'])

    temp = [config.temp0, config.temp1, config.temp2]

    explainer = MyExplainer(config.training_mask, epochs=config.expl_epochs, lr=config.lr, 
                            size_reg=config.size_reg, mask_thr=config.mask_thr, temp=temp, device=device)
    
    explainer.prepare(model)
    graph, grad = explainer.train(train_loader)

    # list_epochs = list(range(config.expl_epochs))
    # figure, axis = plt.subplots(4,1)
    # axis[0].plot(list_epochs, graph['loss'])
    # axis[1].plot(list_epochs, graph['cce_loss'])
    # axis[2].plot(list_epochs, graph['size_loss'])
    # axis[3].plot(list_epochs, grad)
    # plt.show()

    acc, fid, inf, num, auc = run_experiment(explainer, test_subgraph_loader, test_original_loader, config, b_plot)

    return auc, acc, fid, inf, num    