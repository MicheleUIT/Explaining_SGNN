
import os
import torch
from torch_geometric.loader import DataLoader

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
    torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    return

def load_best_model(dataset, model, device):
    checkpoint = torch.load(f"./surrogate/{dataset}/best_model", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()
    
def train_graph(model, dataset, device, epochs=350, lr=0.005, early_stop=20):
    
    # ensure edge_attr is not considered
    
    split_idx = dataset.separate_data(0, fold_idx=0)
    
    train_loader = DataLoader(dataset[split_idx["train"]],
                              batch_size=32, shuffle=True)
    train_loader_eval = DataLoader(dataset[split_idx["train"]],
                              batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset[split_idx["test"]],
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
    model.eval().to(device)
    return model


class MyExplainer():
    def __init__(self, model_to_explain, dataset, epochs=50, lr=0.001, reg_coefs=(0.001, 0.001), gt_size = 6, device='cuda'):
        super().__init__()
        self.model_to_explain = model_to_explain
        self.dataset = dataset
        self.epochs = epochs
        self.lr = lr
        self.reg_coefs = reg_coefs
        self.gt_size = gt_size
        self.device = device
        self.size_reg = reg_coefs[0]
        self.entropy_reg = reg_coefs[1]

        self.explainer_model = nn.Sequential(
            nn.Linear(self.model_to_explain.hidden_dim * 2, 64),
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
            index = torch.nonzero(soft>=0.5).squeeze()
            #index = torch.sort(soft, descending=True)[1][:]
        else:
            soft = logits.sigmoid()
            index = torch.sort(soft, descending=True)[1][:size]
        hard = torch.zeros_like(
                logits).scatter_(-1, index, 1.0) - soft.detach() + soft
        return soft, hard


    def l_loss(self, masked_pred, original_pred, soft):
        size_loss = torch.sum(soft) * self.size_reg
        mask_ent_reg = -soft * torch.log(soft) - (1 - soft) * torch.log(1 - soft)
        mask_ent_loss = self.entropy_reg * torch.mean(mask_ent_reg)
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        return cce_loss + mask_ent_loss + size_loss 

    def _loss(self, masked_pred, original_pred, hard):
        size_loss = torch.sum(hard) * self.size_reg
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        return cce_loss + size_loss 
    
    def train(self):

        self.explainer_model.train()
        self.model_to_explain.eval()
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        c = 0
        bsize= 1
        train_loader = DataLoader(self.dataset,
                              batch_size=bsize, shuffle=True)


        #temp_schedule2 = lambda e: 1*(0.5)**(e/self.epochs)
        #temp_schedule = lambda e: 1*(1)**(e/self.epochs)
        temp_schedule = lambda e: 5 - 4 *(e/self.epochs)
        #temp_schedule2 = lambda e: 1 - 0.5 *(e/self.epochs)
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss_detached = 0
            stability = 0
            size = 0
            #t1 = temp_schedule1(e)
            t = temp_schedule(e)
            #t2 = temp_schedule2(e)
            #t2 = max(0.5, 5e-5*(e+1))
            #t2 = 1
            for data in train_loader:
                #c+=1
                #t = max(0.5, 5e-5*(c))
                data.to(self.device)
                feats = data.x.detach()
                graph = data.edge_index.detach()
                with torch.no_grad():
                    original_pred = self.model_to_explain(feats, graph, data.batch).argmax(dim=-1)
                    embeds = self.model_to_explain.embedding(feats, graph)
                input_expl = self._create_explainer_input(graph, embeds)
                sampling_weights = self.explainer_model(input_expl).squeeze()
                sm, hm = self._sample_graph(sampling_weights, t)
                masked_pred = self.model_to_explain(feats, graph, data.batch, edge_weight=hm)
                loss = self._loss(masked_pred, original_pred, hm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.explainer_model.parameters(), 2.0)
                optimizer.step()
                loss_detached += loss.detach().item()
                stability += (original_pred == masked_pred.argmax(dim=-1)).float().mean()
                size += hm.sum().detach().item() / bsize
            train_loss = loss_detached / len(train_loader)
            stabilities = stability / len(train_loader)
            sizes = size / (len(train_loader) )
            print(f"Epoch: {e}, train_loss: {train_loss:.2f}, stability: {stabilities:.2f}, size: {sizes:.2f}")

    def explain(self):

        self.explainer_model.train()
        self.model_to_explain.eval()
        train_loader = DataLoader(self.dataset, batch_size=1)

        acc=0
        for data in train_loader:
            data.to(self.device)
            feats = data.x.detach()
            graph = data.edge_index.detach()
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph, data.batch).argmax(dim=-1)
                embeds = self.model_to_explain.embedding(feats, graph)
            input_expl = self._create_explainer_input(graph, embeds)
            sampling_weights = self.explainer_model(input_expl).squeeze()

            stability=0
            for i in range(20):
                sm, hm = self._sample_graph(sampling_weights, training=False, size=i)
                masked_pred = self.model_to_explain(feats, graph, data.batch, edge_weight=hm)
                stability += (original_pred == masked_pred.argmax(dim=-1)).float()

            acc+= stability/20
        print(acc/len(train_loader))