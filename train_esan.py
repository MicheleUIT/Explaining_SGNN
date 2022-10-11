import os
import torch
import torch.optim as optim
import random
import wandb
import numpy as np

from tqdm import tqdm
from esan_utils.data import *
from esan_utils.utils import get_data, get_model, SimpleEvaluator


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
                y_pred.append(pred.argmax(dim=-1).detach().cpu())

        all_y_pred.append(torch.cat(y_pred, dim=0).unsqueeze(-1).unsqueeze(-1).numpy())

    y_true = torch.cat(y_true, dim=0).reshape(-1,1).numpy()
    input_dict = {"y_true": y_true, "y_pred": all_y_pred}
    return evaluator.eval(input_dict)


def run(args, device, fold_idx):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_loader, _, valid_loader, _, attributes = get_data(args, fold_idx, device)
    in_dim, out_dim, task_type, eval_metric = attributes

    evaluator = SimpleEvaluator(task_type)

    model = get_model(args, in_dim, out_dim, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    criterion = torch.nn.CrossEntropyLoss()

    
    # If sampling, perform majority voting on the outputs of 5 independent samples
    voting_times = 5 if args.fraction != 1. else 1
    valid_curve = []
    best_val_mae = 1000.0 # arbitrarily large number
    best_val_acc = 0.0
    single=False
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
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
                best_val_acc = valid_perf[eval_metric]
                # print(f"Accuracy improved: {best_val_acc}")
                store_checkpoint(args, model, best_val_acc)

            
    return valid_curve


def main():
    args = {
        'gnn_type': 'pgegin',
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
        'policy': 'node_deleted',
        'num_hops': 2,
        'num_workers': 0,
        'model': 'deepsets',
        'fraction': 0.1,
        'random_ratio': 0,
        'drop_ratio': 0.5,
        'seed': 0
        }
    wandb.init(config=args)
    args = wandb.config

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
        print(f"Fold: {fold_idx}")        
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