
import os
import torch
#from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np


def index_edge(graph, pair):
    return torch.where((graph.T == pair).all(dim=1))[0]


def evaluate(out, labels):
    preds = out.argmax(dim=1)
    acc = (preds == labels).float().mean()
    return acc

def store_checkpoint(dataset, model, train_acc, val_acc):
    save_dir = f"./surrogate/{dataset}/"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint, os.path.join(save_dir, f"chkpt"))
    return

def load_best_model(dataset, model, device):
    checkpoint = torch.load(f"./surrogate/{dataset}/chkpt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()
    

def store_model(dataset, model):
    save_dir = f"./surrogate/{dataset}/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(model, os.path.join(save_dir, f"model"))
    return

def load_model(dataset, device):
    model = torch.load(f"./surrogate/{dataset}/model", map_location=device)
    return model.eval()


def train_graph(model, dataset, device, epochs=350, lr=0.005, early_stop=20):
    
    # ensure edge_attr is not considered
    
    split_idx = dataset.separate_data(0, fold_idx=0)
    
    train_loader = DataLoader(dataset[split_idx["train"]],
                              batch_size=32, shuffle=True)
    train_loader_eval = DataLoader(dataset[split_idx["train"]],
                              batch_size=32, shuffle=True)

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
   
    for epoch in range(0, epochs):
        model.train()
        train_sum = 0
        loss_detached = 0
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_detached += loss.detach()
            train_sum += evaluate(out, data.y)

        train_acc = train_sum / len(train_loader)
        train_loss = loss_detached / len(train_loader)
        
        # Evaluate train
        model.eval()
        with torch.no_grad():
            eval_data = next(iter(train_loader_eval)).to(device)
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            val_acc = evaluate(out, eval_data.y)

        print(f"Epoch: {epoch}, eval_acc: {val_acc:.3f}, train_loss: {train_loss:.3f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(dataset.name, model, train_acc, val_acc)

        # Early stopping
        if epoch - best_epoch > early_stop:
            break

    model = load_best_model(dataset.name, model, device)

    store_model(dataset.name, model)
    model.eval().to(device)
    return model


class MyExplainer():
    def __init__(self, dataset, epochs=50, lr=0.003, reg_coefs=(0.0003, 0.1), gt_size = 6, device='cuda'):
        super().__init__()
        self.dataset = dataset
        self.epochs = epochs
        self.lr = lr
        self.reg_coefs = reg_coefs
        self.gt_size = gt_size
        self.device = device
        self.size_reg = reg_coefs[0]
        self.entropy_reg = reg_coefs[1]
        self.temp= (5,1)

    def prepare(self, model):
        self.model = model
        self.explainer = nn.Sequential(
            nn.Linear(self.model.hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

    def _create_explainer_input(self, graph, embeds):
        row_embeds = embeds[graph[0]]
        col_embeds = embeds[graph[1]]
        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def _sample_graph(self, logits, training=True, t=1, t2=1, size=10):
        if training:
            gumbels = -torch.empty_like(logits).exponential_().log()* t2
            gumbels = (logits + gumbels) / t
            soft = gumbels.sigmoid()
            index = torch.nonzero(soft>0.5).squeeze()
        else:
            soft = logits.sigmoid()
            index = torch.sort(soft, descending=True)[1][:size]
        hard = torch.zeros_like(
                logits).scatter_(-1, index, 1.0) - soft.detach() + soft
        return soft, hard

    def _loss(self, masked_pred, original_pred, hard):
        size_loss = (torch.sum(hard) - 2).abs() * self.size_reg
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        return cce_loss + size_loss 
    
    def train(self):

        self.explainer.train()
        self.model.eval()
        optimizer = Adam(self.explainer.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))
        bsize= 16
        train_loader = DataLoader(self.dataset,
                              batch_size=bsize, shuffle=True)

        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss_detached = 0
            stability = 0
            size = 0
            t = temp_schedule(e)
            for data in train_loader:
                data.to(self.device)
                feats = data.x.detach()
                graph = data.edge_index.detach()
                with torch.no_grad():
                    original_pred = self.model(feats, graph, data.batch).argmax(dim=-1)
                    embeds = self.model.embedding(feats, graph)
                input_expl = self._create_explainer_input(graph, embeds)
                sampling_weights = self.explainer(input_expl).squeeze()
                sm, hm = self._sample_graph(sampling_weights, t)
                masked_pred = self.model(feats, graph, data.batch, edge_weight=sm)
                loss = self._loss(masked_pred, original_pred, hm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.explainer.parameters(), 2.0)
                optimizer.step()
                loss_detached += loss.detach().item()
                stability += (original_pred == masked_pred.argmax(dim=-1)).float().mean()
                size += hm.sum().detach().item() / bsize
            train_loss = loss_detached / len(train_loader)
            stabilities = stability / len(train_loader)
            sizes = size / (len(train_loader) )
            print(f"Epoch: {e}, train_loss: {train_loss:.2f}, stability: {stabilities:.2f}, size: {sizes:.2f}")

    def explain(self):

        self.explainer.train()
        self.model.eval()
        train_loader = DataLoader(self.dataset, batch_size=1)

        acc=0
        for data in train_loader:
            data.to(self.device)
            feats = data.x.detach()
            graph = data.edge_index.detach()
            with torch.no_grad():
                original_pred = self.model(feats, graph, data.batch).argmax(dim=-1)
                embeds = self.model.embedding(feats, graph)
            input_expl = self._create_explainer_input(graph, embeds)
            sampling_weights = self.explainer(input_expl).squeeze()

            stability=0
            size = 0
            #for i in range(20):
            #    _, hm = self._sample_graph(sampling_weights, training=False, size=i)
            #    masked_pred = self.model(feats, graph, data.batch, edge_weight=hm)
            #    stability += (original_pred == masked_pred.argmax(dim=-1)).float()

            for i in range(20):
                _, hm = self._sample_graph(sampling_weights, t=1)
                masked_pred = self.model(feats, graph, data.batch, edge_weight=hm)
                stability += (original_pred == masked_pred.argmax(dim=-1)).float()
                size+= hm.sum()


            print("graph stability:", stability.item()/20, "and size: ", size.item()/20)    
            acc+= stability/20
            
        print("Explanation:",(acc/len(train_loader)).item())


    def save(self, dataset):
        save_dir = f"./explainer/{dataset}/"
        checkpoint = {'explainer_state_dict': self.explainer.state_dict()}
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        torch.save(checkpoint, os.path.join(save_dir, f"chkpt"))
        return

    def load(self, dataset, device):
        checkpoint = torch.load(f"./explainer/{dataset}/chkpt", map_location=device)
        self.prepare(load_model(dataset, device=device))
        self.explainer.load_state_dict(checkpoint['explainer_state_dict'])
        self.explainer.eval()
        return 