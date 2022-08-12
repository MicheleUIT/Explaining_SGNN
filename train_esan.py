import os
import torch
import torch.optim as optim
import random
import wandb
import numpy as np

from data import *
from utils import get_data, get_model, SimpleEvaluator, NonBinaryEvaluator, Evaluator


def store_checkpoint(args, model, best_val):
    save_dir = f"./checkpoints/{args.dataset}/{args.model}/{args.policy}/{args.gnn_type}_{args.seed}/"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'best_val': best_val}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    pass


def load_best_model(args, model, device):
    checkpoint = torch.load(f"./checkpoints/{args.dataset}/{args.model}/{args.policy}/{args.gnn_type}_{args.seed}/best_model", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def train(model, device, loader, optimizer, criterion, single=False):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            if single:
                pred = model.single(batch)
            else:
                pred = model(batch)
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y

            y = batch.y.view(pred.shape).to(torch.float32) if pred.size(-1) == 1 else batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])

            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator, voting_times=1, single=False):
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
                    if single:
                        pred = model.single(batch)
                    else:
                        pred = model(batch)

                y = batch.y.view(pred.shape) if pred.size(-1) == 1 else batch.y
                y_true.append(y.detach().cpu())
                y_pred.append(pred.detach().cpu())

        all_y_pred.append(torch.cat(y_pred, dim=0).unsqueeze(-1).numpy())

    y_true = torch.cat(y_true, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": all_y_pred}
    return evaluator.eval(input_dict)


def run(args, device, fold_idx):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_loader, _, valid_loader, _, attributes = get_data(args, fold_idx, device)
    in_dim, out_dim, task_type, eval_metric = attributes

    if 'ogb' in args.dataset:
        evaluator = Evaluator(args.dataset)
    else:
        evaluator = SimpleEvaluator(task_type) if args.dataset != "IMDB-MULTI" \
                                                  and args.dataset != "CSL" else NonBinaryEvaluator(out_dim)

    model = get_model(args, in_dim, out_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if 'ogb' in args.dataset:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    criterion = torch.nn.BCEWithLogitsLoss() if args.dataset != "IMDB-MULTI" \
                                                and args.dataset != "CSL" else torch.nn.CrossEntropyLoss()

    
    # If sampling, perform majority voting on the outputs of 5 independent samples
    voting_times = 5 if args.fraction != 1. else 1
    valid_curve = []
    best_val_mae = 1000.0 # arbitrarily large number, is there a better way?
    best_val_acc = 0.0
    single=False
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, single)
        if scheduler is not None:
            scheduler.step()

        valid_perf = eval(model, device, valid_loader, evaluator, voting_times, single)
        valid_curve.append(valid_perf[eval_metric])
        wandb.log({"Val_curve": valid_perf[eval_metric]})
        if eval_metric == 'mae':
            if valid_perf[eval_metric] < best_val_mae:  # New best results
                # print("Mean absolute errore improved")
                best_val_mae = valid_perf[eval_metric]
                store_checkpoint(args, model, best_val_mae)
            
        elif eval_metric == 'acc':
            if valid_perf[eval_metric] > best_val_acc:  # New best results
                # print("Accuracy improved")
                best_val_acc = valid_perf[eval_metric]
                store_checkpoint(args, model, best_val_acc)

            
    return valid_curve


def main():
    # Training settings
    # parser = argparse.ArgumentParser(description='GNN baselines with Pytorch Geometrics')
    # parser.add_argument('--device', type=int, default=0,
    #                     help='which gpu to use if any (default: 0)')
    # parser.add_argument('--gnn_type', type=str,
    #                     help='Type of convolution {gin, originalgin, zincgin, graphconv}')
    # parser.add_argument('--random_ratio', type=float, default=0.,
    #                     help='Number of random features, > 0 only for RNI')
    # parser.add_argument('--model', type=str,
    #                     help='Type of model {deepsets, dss, gnn}')
    # parser.add_argument('--drop_ratio', type=float, default=0.5,
    #                     help='dropout ratio (default: 0.5)')
    # parser.add_argument('--num_layer', type=int, default=5,
    #                     help='number of GNN message passing layers (default: 5)')
    # parser.add_argument('--channels', type=str,
    #                     help='String with dimension of each DS layer, separated by "-"'
    #                          '(considered only if args.model is deepsets)')
    # parser.add_argument('--emb_dim', type=int, default=300,
    #                     help='dimensionality of hidden units in GNNs (default: 300)')
    # parser.add_argument('--jk', type=str, default="last",
    #                     help='JK strategy, either last or concat (default: last)')
    # parser.add_argument('--batch_size', type=int, default=32,
    #                     help='input batch size for training (default: 32)')
    # parser.add_argument('--learning_rate', type=float, default=0.01,
    #                     help='learning rate for training (default: 0.01)')
    # parser.add_argument('--decay_rate', type=float, default=0.5,
    #                     help='decay rate for training (default: 0.5)')
    # parser.add_argument('--decay_step', type=int, default=50,
    #                     help='decay step for training (default: 50)')
    # parser.add_argument('--epochs', type=int, default=100,
    #                     help='number of epochs to train (default: 100)')
    # parser.add_argument('--num_workers', type=int, default=0,
    #                     help='number of workers (default: 0)')
    # parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
    #                     help='dataset name (default: ogbg-molhiv)')
    # parser.add_argument('--policy', type=str, default="edge_deleted",
    #                     help='Subgraph selection policy in {edge_deleted, node_deleted, ego_nets}'
    #                          ' (default: edge_deleted)')
    # parser.add_argument('--num_hops', type=int, default=2,
    #                     help='Depth of the ego net if policy is ego_nets (default: 2)')
    # parser.add_argument('--seed', type=int, default=0,
    #                     help='random seed (default: 0)')
    # parser.add_argument('--fraction', type=float, default=1.0,
    #                     help='Fraction of subsampled subgraphs (1.0 means full bag aka no sampling)')
    # parser.add_argument('--patience', type=int, default=20,
    #                     help='patience (default: 20)')
    # parser.add_argument('--test', action='store_true',
    #                     help='quick test')
    # parser.add_argument('--filename', type=str, default="",
    #                     help='filename to output result (default: )')

    # # new args
    # parser.add_argument('--ex_mask', type=str, default="hard",
    #                     help='mask to use for prediction')
    # parser.add_argument('--ex_epochs', type=int, default=20,
    #                     help='explainer epochs (default: 20)')
    # parser.add_argument('--ex_lr', type=float, default=0.01,
    #                     help='explainer learning rate')
    # parser.add_argument('--ex_t1', type=float, default=5,
    #                     help='explainer starting temperature')
    # parser.add_argument('--ex_t2', type=float, default=1,
    #                     help='explainer ending temperature')
    # parser.add_argument('--ex_t3', type=float, default=5,
    #                     help='gumbel noise scaling')
    # parser.add_argument('--ex_size', type=float, default=0.1,
    #                     help='size regularization')
    # parser.add_argument('--ex_noise', type=bool, default=False,
    #                     help='noisy graph inference')

    # args = parser.parse_args()
    args = {
        'gnn_type': 'originalgin',
        'num_layer': 4,
        'emb_dim': 32,
        'batch_size': 32,
        'learning_rate': 0.005,
        'decay_rate': 0.5,
        'decay_step': 50,
        'epochs': 350,
        'dataset': 'Mutagenicity',
        'jk': 'concat',
        'drop_ratio': 0.,
        'channels': '32-32',
        'policy': 'ego_nets',
        'num_hops': 2,
        'num_workers': 0,
        'model': 'dss',
        'fraction': 0.1,
        'random_ratio': 0,
        'drop_ratio': 0.5,
        'seed': 0
        }
    wandb.init(project="expl_esan",
            entity="tromso",
            config=args)
    args = wandb.config


    # args.channels = list(map(int, args.channels.split("-")))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if False:
        n_folds = 1
    else:
        n_folds = 10

    curve_folds = []
    fold_idx = 0
    while len(curve_folds) < n_folds:         
        results = run(args, device, fold_idx)
        curve_folds.append(results)
        fold_idx += 1

    valid_curve_folds = np.array(curve_folds)
 
    valid_curve = np.mean(valid_curve_folds, 0)
    valid_accs_std = np.std(valid_curve_folds, 0)
    best_val_epoch = np.argmax(valid_curve)
    print(best_val_epoch)

    wandb.log({'Best_epoch': best_val_epoch, 'Val': valid_curve[best_val_epoch], 'Val_std': valid_accs_std[best_val_epoch]})


if __name__ == "__main__":
    main()