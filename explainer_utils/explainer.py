import torch
import wandb
import torch_scatter

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.utils import to_undirected
from explainer_utils.auc import evaluation_auc


class MyExplainer():
    def __init__(self, training_mask='hard', epochs=30, lr=0.003, temp=(5.0, 1.0, 1.0), size_reg=.5, noise=False, mask_thr=0.5, device='cuda'):
        super().__init__()

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.device = device
        self.size_reg = size_reg
        self.mask_thr = mask_thr
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
                    original_pred = self.model(data).argmax(dim=-1)
                    embeds = self.model.embedding(data)
                input_expl = self._create_explainer_input(data.edge_index, embeds)
                sampling_weights = self.explainer(input_expl).squeeze()
                sm, hm = self._sample_graph(data.edge_index, sampling_weights, t)
                # Use soft or hard mask for training
                if self.training_mask == 'hard':
                    edge_weight=hm
                else:
                    edge_weight=sm
                masked_pred = self.model(data, edge_weight=edge_weight)
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

            # wandb.log({"Ex_loss": train_loss, "Ex_stability": stabilities, "Ex_size":sizes})

    
    def explain(self, test_subgraph_loader, test_original_loader):
        self.explainer.eval()
        self.model.eval()

        accs = []
        fids = []
        infs = []
        sums = []

        explanations = []
        ground_truths = []

        for sub, orig_graph in tqdm(zip(test_subgraph_loader, test_original_loader)):
            sub.to(self.device)

            embeds = self.model.embedding(sub).detach()

            with torch.no_grad():
                edge_index, subgraph_node_idx, batch, num_nodes_per_subgraph = sub.edge_index, sub.subgraph_node_idx, sub.batch, sub.num_nodes_per_subgraph

            input_expl = self._create_explainer_input(edge_index, embeds)
            sampling_weights = self.explainer(input_expl).squeeze()
            soft, _ = self._sample_graph(edge_index, sampling_weights, training=False)

            tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype).detach(),
                                torch.cumsum(num_nodes_per_subgraph, dim=0)])
            graph_offset = tmp[batch]
            node_idx = graph_offset + subgraph_node_idx

            edge_index_new = torch.zeros_like(edge_index).detach().cpu()
            edge_index_new[0,:] = node_idx[edge_index[0,:]]
            edge_index_new[1,:] = node_idx[edge_index[1,:]]

            orig_edges = orig_graph.edge_index.detach().cpu()
            orig_mask = torch.zeros_like(orig_graph.edge_gt)

            for i in range(len(orig_mask)):
                indices = torch.where((edge_index_new.T==orig_edges.T[i]).all(dim=1),1,0)
                orig_mask[i] = torch_scatter.scatter(soft.detach().cpu(),indices,dim=0,reduce="sum")[1]
            
            orig_mask = orig_mask / orig_mask.max()
            
            index = torch.nonzero(soft>=self.mask_thr).squeeze()
            hard = torch.zeros_like(soft).scatter_(-1, index, 1.0)

            with torch.no_grad():
                real_label = sub.y.cpu()
                pred_label = self.model(sub).argmax(dim=-1).cpu()
                fid_label = self.model(sub, edge_weight=1-hard).argmax(dim=-1).cpu()
                inf_label = self.model(sub, edge_weight=hard).argmax(dim=-1).cpu()
            
            orig_index = torch.nonzero(orig_mask>=self.mask_thr).squeeze()
            orig_hard = torch.zeros_like(orig_mask).scatter_(-1, orig_index, 1.0)

            explanations.append((orig_edges,orig_mask,orig_hard))
            ground_truths.append(orig_graph.edge_gt)

            acc = int(pred_label == real_label)
            accs.append(acc)
            fids.append(acc - int(fid_label == real_label))
            infs.append(acc - int(inf_label == real_label))
            sums.append(orig_hard.detach().sum().cpu()/len(orig_hard))

        auc = evaluation_auc(explanations, ground_truths)

        return accs, fids, infs, sums, auc, explanations
