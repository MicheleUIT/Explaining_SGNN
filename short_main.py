import argparse
import multiprocessing as mp
import os
import random

import numpy as np
import torch
import torch.optim as optim
#from ogb.graphproppred import Evaluator

# noinspection PyUnresolvedReferences
from data import SubgraphData
from utils import get_data, get_model, SimpleEvaluator, NonBinaryEvaluator, Evaluator
import torch_geometric

torch.set_num_threads(1)


def train(model, device, loader, optimizer, criterion, epoch, fold_idx):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

            y = batch.y.view(pred.shape).to(torch.float32) if pred.size(-1) == 1 else batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])

            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator, voting_times=1):
    model.eval()

    all_y_pred = []
    for i in range(voting_times):
        y_true = []
        y_pred = []

        for step, batch in enumerate(loader):
            batch = batch.to(device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)

                y = batch.y.view(pred.shape) if pred.size(-1) == 1 else batch.y
                y_true.append(y.detach().cpu())
                y_pred.append(pred.detach().cpu())

        all_y_pred.append(torch.cat(y_pred, dim=0).unsqueeze(-1).numpy())

    y_true = torch.cat(y_true, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": all_y_pred}
    return evaluator.eval(input_dict)


def run(args, device, fold_idx, results_queue):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_loader, train_loader_eval, valid_loader, test_loader, attributes = get_data(args, fold_idx)
    in_dim, out_dim, task_type, eval_metric = attributes

    evaluator = SimpleEvaluator(task_type) if args.dataset != "IMDB-MULTI" \
                                            else NonBinaryEvaluator(out_dim)

    model = get_model(args, in_dim, out_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    if "classification" in task_type:
        criterion = torch.nn.BCEWithLogitsLoss() if args.dataset != "IMDB-MULTI" \
                                                    else torch.nn.CrossEntropyLoss()

    # If sampling, perform majority voting on the outputs of 5 independent samples
    voting_times = 5 if args.fraction != 1. else 1

    valid_curve = []

    for epoch in range(1, args.epochs + 1):

        train(model, device, train_loader, optimizer, criterion, epoch=epoch, fold_idx=fold_idx)
        # Only valid_perf is used for TUD
        valid_perf = eval(model, device, valid_loader, evaluator, voting_times)
        scheduler.step()
        valid_curve.append(valid_perf[eval_metric])

    results_queue.put((train_curve, valid_curve, test_curve))
    return


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default='originalgin',
                        help='Type of convolution {gin, originalgin, zincgin, graphconv}')
    parser.add_argument('--random_ratio', type=float, default=0.,
                        help='Number of random features, > 0 only for RNI')
    parser.add_argument('--model', type=str, default='deepsets',
                        help='Type of model {deepsets, dss, gnn}')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='dropout ratio (defauGNNlt: 0.0)')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers (default: 4)')
    parser.add_argument('--channels', type=str,default='32-32',
                        help='String with dimension of each DS layer, separated by "-"'
                             '(considered only if args.model is deepsets)')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--jk', type=str, default="concat",
                        help='JK strategy, either last or concat (default: last)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate for training (default: 0.005)')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate for training (default: 0.5)')
    parser.add_argument('--decay_step', type=int, default=50,
                        help='decay step for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--policy', type=str, default="edge_deleted",
                        help='Subgraph selection policy in {edge_deleted, node_deleted, ego_nets}'
                             ' (default: edge_deleted)')
    parser.add_argument('--num_hops', type=int, default=2,
                        help='Depth of the ego net if policy is ego_nets (default: 2)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of subsampled subgraphs (1.0 means full bag aka no sampling)')
    parser.add_argument('--patience', type=int, default=20,
                        help='patience (default: 20)')
    parser.add_argument('--test', action='store_true',
                        help='quick test')

    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')

    args = parser.parse_args()

    args.channels = list(map(int, args.channels.split("-")))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mp.set_start_method('spawn')
    n_folds = 10
    if n_folds > 1:
        if args.dataset == 'PROTEINS':
            num_proc = 2
        else:
            num_proc = 3 if args.batch_size == 128 and args.dataset != 'MUTAG' and args.dataset != 'PTC' else 5

    num_free = num_proc
    results_queue = mp.Queue()

    curve_folds = []
    fold_idx = 0

    if args.test:
        run(args, device, fold_idx, results_queue)
        exit()

    while len(curve_folds) < n_folds:
        if num_free > 0 and fold_idx < n_folds:
            p = mp.Process(
                target=run, args=(args, device, fold_idx, results_queue)
            )
            fold_idx += 1
            num_free -= 1
            p.start()
        else:
            curve_folds.append(results_queue.get())
            num_free += 1

    valid_curve_folds = np.array([l[1] for l in curve_folds])

    valid_curve = np.mean(valid_curve_folds, 0)
    valid_accs_std = np.std(valid_curve_folds, 0)


    task_type = 'classification'

    best_val_epoch = np.argmax(valid_curve)
    best_train = max(train_curve)

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Val std': valid_accs_std[best_val_epoch],
                    'BestTrain': best_train}, args.filename)

    print({'Val', valid_curve[best_val_epoch], 'Val std', valid_accs_std[best_val_epoch],
                    'BestTrain', best_train})
if __name__ == "__main__":
    main()
