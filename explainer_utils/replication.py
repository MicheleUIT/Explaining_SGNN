import numpy as np

from explainer_utils.explainer import MyExplainer
from data import MutagGTDataset, filter_gt
from torch_geometric.data import DataLoader



def run_experiment(explainer, test_subgraph_loader, test_original_loader):
    
    accs, fids, infs, sums, auc = explainer.explain(test_subgraph_loader, test_original_loader)

    acc = np.asarray(accs).mean()
    fid = np.asarray(fids).mean()
    inf = np.asarray(infs).mean()
    n   = np.asarray(sums).mean()
    
    return acc, fid, inf, n, auc


def explain(model, dataset, args, config, device='cuda'):
    
    orig_dataset = MutagGTDataset(root="dataset/prefiltered/" + "original",
                            name=args.dataset,
                            pre_transform=None,
                            pre_filter=filter_gt
                            )
    
    train_loader = DataLoader(dataset, args.batch_size, shuffle=True, follow_batch=['subgraph_idx', 'original_x'])
    test_subgraph_loader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=['subgraph_idx', 'original_x']) 
    test_original_loader = DataLoader(orig_dataset, batch_size=1, shuffle=False, follow_batch=['subgraph_idx', 'original_x']) 

    explainer = MyExplainer(config.training_mask, epochs=config.expl_epochs, lr=config.lr, 
                            size_reg=config.size_reg, mask_thr=config.mask_thr, temp=config.temp, device=device)
    
    explainer.prepare(model)
    explainer.train(train_loader)

    acc, fid, inf, num, auc = run_experiment(explainer, test_subgraph_loader, test_original_loader)
          
    return auc, acc, fid, inf, num