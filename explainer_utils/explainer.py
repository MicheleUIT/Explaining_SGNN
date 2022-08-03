import torch
import wandb
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.utils import to_undirected


class MyExplainer():
    def __init__(self, training_mask='hard', epochs=30, lr=0.003, temp=(5.0, 1.0, 1.0), size_reg=.5, noise=False, device='cuda'):
        super().__init__()

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.device = device
        self.size_reg = size_reg
        self.mask_thr = 0.5
        self.training_mask = training_mask
        self.noise = noise

    def _create_explainer_input(self, graph, embeds):
        row_embeds = embeds[graph[0]]
        col_embeds = embeds[graph[1]]
        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl


    def prepare(self, model):
        self.model = model
        self.explainer = nn.Sequential(
            nn.Linear(self.model.emb_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)


    def _sample_graph(self, graph, logits, training=True, temperature: float = 1):
        if training:
            gumbels_n = -torch.empty_like(logits).exponential_().log()/self.temp[2]
            gumbels = (logits + gumbels_n) / temperature  
            soft = gumbels.sigmoid()
        else:
            soft = logits.sigmoid()
        _, und_soft = to_undirected(edge_index=graph, edge_attr=soft, reduce='max')
        index = torch.nonzero(und_soft>=self.mask_thr).squeeze()
        hard = torch.zeros_like(logits).scatter_(-1, index, 1.0) - und_soft.detach() + und_soft
    
        return und_soft, hard

    def _sample_k_graph(self, graph, logits, training=True, temperature: float = 1, size: int = 10):
        if training:
            gumbels_n = -torch.empty_like(logits).exponential_().log()/self.temp[2]
            gumbels = (logits + gumbels_n) / temperature  
            soft = gumbels.sigmoid()
        else:
            soft = logits.sigmoid()
        _, und_soft = to_undirected(edge_index=graph, edge_attr=soft, reduce='max')
        index = torch.sort(und_soft, descending=True)[1][:size]
        hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return hard


    def _loss(self, masked_pred, original_pred, hard):
        size_loss = torch.abs(torch.sum(hard) - (hard.size(0)*5) //100) * self.size_reg
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        return cce_loss + size_loss

    
    def train(self, data_loader):
        self.explainer.train()
        self.model.eval()
        optimizer = Adam(self.explainer.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss_detached = 0
            stability = 0
            size = 0
            t = temp_schedule(e)
            for data in data_loader:
                data.to(self.device)
                with torch.no_grad():
                    original_pred = self.model.single(data).argmax(dim=-1) # perche' single?
                    embeds = self.model.embedding(data)
                input_expl = self._create_explainer_input(data.edge_index, embeds)
                sampling_weights = self.explainer(input_expl).squeeze()
                sm, hm = self._sample_graph(data.edge_index, sampling_weights, t)
                # Use soft or hard mask for training
                if self.training_mask == 'hard':
                    edge_weight=hm
                else:
                    edge_weight=sm
                masked_pred = self.model.single(data, edge_weight=edge_weight)
                loss = self._loss(masked_pred, original_pred, hm)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.explainer.parameters(), 2.0)
                optimizer.step()
                loss_detached += loss.detach().item()
                stability += (original_pred == masked_pred.argmax(dim=-1)).float().mean()
                size += hm.sum().detach().item() / data_loader.batch_size
            train_loss = loss_detached / len(data_loader)
            stabilities = stability / len(data_loader)
            sizes = size / (len(data_loader))

            wandb.log({"Ex_loss": train_loss, "Ex_stability": stabilities, "Ex_size":sizes})

    
    def explain(self, data):
        self.explainer.eval()
        self.model.eval()
        data.to(self.device)

        embeds = self.model.embedding(data).detach()

        input_expl = self._create_explainer_input(data.edge_index, embeds)
        sampling_weights = self.explainer(input_expl).squeeze()
        soft, hard = self._sample_graph(data.edge_index, sampling_weights, training=False, mask_thr=self.mask_thr)        
        
        with torch.no_grad():
            pred_label = self.model.single(data).argmax(dim=-1).detach().cpu()
            fid_label = self.model.single(data, edge_weight=1-hard).argmax(dim=-1).detach().cpu()
            inf_label = self.model.single(data, edge_weight=hard).argmax(dim=-1).detach().cpu()
        
        real_label = data.y
        acc = int(pred_label == real_label)
        fid = acc - int(fid_label == real_label)
        inf = acc - int(inf_label == real_label)

        return data.edge_index.cpu(), soft.detach().cpu(), [acc, fid, inf, hard.detach().sum().cpu()]
