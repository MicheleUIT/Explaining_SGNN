import numpy as np
import matplotlib.pyplot as plt

from explainer_utils.explainer import MyExplainer
from data import MutagGTDataset, BA2GTDataset, filter_gt
from torch_geometric.data import DataLoader
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


def explain(model, dataset, config, b_plot = False, device='cuda'):
    
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
    
    # dataset.data.edge_attr = None
    # split_idx1 = dataset.separate_data(0, fold_idx=0)

    # orig_dataset.data.edge_attr = None
    # split_idx2 = orig_dataset.separate_data(0, fold_idx=0)
    
    train_loader = DataLoader(dataset, config.batch_size, shuffle=True, follow_batch=['subgraph_idx', 'original_x'])
    test_subgraph_loader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=['subgraph_idx', 'original_x']) 
    test_original_loader = DataLoader(orig_dataset, batch_size=1, shuffle=False, follow_batch=['subgraph_idx', 'original_x'])

    temp = [config.temp0, config.temp1, config.temp2]

    explainer = MyExplainer(config.training_mask, epochs=config.expl_epochs, lr=config.lr, 
                            size_reg=config.size_reg, mask_thr=config.mask_thr, temp=temp, device=device)
    
    explainer.prepare(model)
    loss, grad = explainer.train(train_loader)

    list_epochs = list(range(config.expl_epochs))
    figure, axis = plt.subplots(2,1)
    axis[0].plot(list_epochs, loss)
    axis[1].plot(list_epochs, grad)
    plt.show()

    acc, fid, inf, num, auc = run_experiment(explainer, test_subgraph_loader, test_original_loader, config, b_plot)

    return auc, acc, fid, inf, num    