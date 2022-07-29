import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse 
import logging
import math
import os
import os.path as osp
import random
import shutil
from typing import Optional, Union, Tuple, Callable, List

import networkx as nx
import numpy as np
import torch
import tqdm
import glob
import torch.nn.functional as F
#from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch_geometric.data import Data, Batch, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import TUDataset as TUDataset_
from torch_geometric.transforms import OneHotDegree, Constant
from torch_geometric.utils import to_undirected, k_hop_subgraph, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.io.tu import read_file, cat
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

from csl_data import MyGNNBenchmarkDataset
from gnn_rni_data import PlanarSATPairsDataset

from os.path import exists


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('The number of nodes in your data object can only be inferred')


logging.getLogger().addFilter(NoParsingFilter())


class MutagGTDataset(InMemoryDataset):

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None, # filtrare solo i grafi mutagenici come in pge?
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
    
    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    # different from parent class
    def download(self):
        # url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = "dataset/" + self.name + ".zip"
        # path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        # os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    # ASSUMPTION: node_idx features for ego_nets_plus are prepended
    @property
    def num_node_labels(self):

        if self.data.x is None:
            return 0
        num_added = 2 if isinstance(self.pre_transform, EgoNets) and self.pre_transform.add_node_idx else 0
        for i in range(self.data.x.size(1) - num_added):
            x = self.data.x[:, i + num_added:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0
    
    @property
    def num_tasks(self):
        return 1

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'
    
    def separate_data(self, seed, fold_idx):

        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test

        assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        labels = self.data.y.cpu().numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return {'train': torch.tensor(train_idx), 'valid': torch.tensor(test_idx), 'test': torch.tensor(test_idx)}


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
    x = cat([node_attributes, node_labels])

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    edge_gt = None
    if 'edge_gt' in names:
        edge_gt = read_file(folder, prefix, 'edge_gt')

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    _, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_gt = remove_self_loops(edge_index, edge_gt)
    # edge_index, edge_labels and edge_gt undirected
    _, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)
    edge_index, edge_gt = coalesce(edge_index, edge_gt, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_gt=edge_gt)
    data, slices = split(data, batch)

    return data, slices


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.edge_gt is not None:
        slices['edge_gt'] = edge_slice

    return data, slices


class TUDataset(TUDataset_):
    def __init__(self, root: str, name: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):

        super().__init__(root, name, transform, pre_transform, pre_filter,
                         use_node_attr, use_edge_attr, cleaned)
        
        if self.name =="PROTEINS":
            self.data.original_x = self.data.original_x[:, 1:]

    @property
    def num_tasks(self):
        return 1 if self.name != "IMDB-MULTI" else 3

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'

    def download(self):
        super().download()

    def process(self):
        super().process()

    # ASSUMPTION: node_idx features for ego_nets_plus are prepended
    @property
    def num_node_labels(self):

        if self.data.x is None:
            return 0
        num_added = 2 if isinstance(self.pre_transform, EgoNets) and self.pre_transform.add_node_idx else 0
        for i in range(self.data.x.size(1) - num_added):
            x = self.data.x[:, i + num_added:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    def separate_data(self, seed, fold_idx):

        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test

        assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        labels = self.data.y.cpu().numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return {'train': torch.tensor(train_idx), 'valid': torch.tensor(test_idx), 'test': torch.tensor(test_idx)}
    

def to_undirected(edge_index: Tensor, edge_attr: Optional[Tensor] = None,
                  num_nodes: Optional[int] = None,
                  reduce: str = "add") -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features. (default: :obj:`"add"`)
    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :class:`Tensor`)
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes, reduce)

    if edge_attr is None:
        return edge_index
    else:
        return edge_index, edge_attr


class Sampler:
    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, data):
        count = math.ceil(self.fraction * len(data.subgraphs))
        sampled_subgraphs = random.sample(data.subgraphs, count)

        batch = Batch.from_data_list(sampled_subgraphs)
        return SubgraphData(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                            subgraph_batch=batch.batch,
                            y=data.y, subgraph_idx=batch.subgraph_idx, subgraph_node_idx=batch.subgraph_node_idx,
                            num_subgraphs=len(sampled_subgraphs), num_nodes_per_subgraph=data.num_nodes,
                            original_x=data.x, original_edge_index=data.edge_index, original_edge_attr=data.edge_attr)


ORIG_EDGE_INDEX_KEY = 'original_edge_index'


class SubgraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == ORIG_EDGE_INDEX_KEY:
            return self.num_nodes_per_subgraph
        else:
            return super().__inc__(key, value)


def preprocess(dataset, transform):
    def unbatch_subgraphs(data):
        subgraphs = []
        num_nodes = data.num_nodes_per_subgraph.item()
        for i in range(data.num_subgraphs):
            edge_index, edge_attr = subgraph(torch.arange(num_nodes) + (i * num_nodes),
                                             data.edge_index, data.edge_attr,
                                             relabel_nodes=False, num_nodes=data.x.size(0))
            subgraphs.append(
                Data(
                    x=data.x[i * num_nodes: (i + 1) * num_nodes, :], edge_index=edge_index - (i * num_nodes),
                    edge_attr=edge_attr,
                    subgraph_idx=torch.tensor(0), subgraph_node_idx=torch.arange(num_nodes),
                    num_nodes=num_nodes,
                )
            )

        original_edge_attr = data.original_edge_attr if data.edge_attr is not None else data.edge_attr
        return Data(x=subgraphs[0].x, edge_index=data.original_edge_index, edge_attr=original_edge_attr, y=data.y,
                    subgraphs=subgraphs)

    data_list = [unbatch_subgraphs(data) for data in dataset]

    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)
    dataset.transform = transform
    return dataset


class Graph2Subgraph:
    def __init__(self, process_subgraphs=lambda x: x, pbar=None):
        self.process_subgraphs = process_subgraphs
        self.pbar = pbar


    def __call__(self, data):
        assert data.is_undirected()

        subgraphs = self.to_subgraphs(data)
        subgraphs = [self.process_subgraphs(s) for s in subgraphs]

        batch = Batch.from_data_list(subgraphs)
        if self.pbar is not None: next(self.pbar)

        return SubgraphData(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                            subgraph_batch=batch.batch,
                            y=data.y, subgraph_idx=batch.subgraph_idx, subgraph_node_idx=batch.subgraph_node_idx,
                            num_subgraphs=len(subgraphs), num_nodes_per_subgraph=data.num_nodes,
                            original_x=data.x, original_edge_index=data.edge_index, original_edge_attr=data.edge_attr)


    def to_subgraphs(self, data):
        raise NotImplementedError




class EdgeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data):
        # remove one of the bidirectional index
        if data.edge_attr is not None and len(data.edge_attr.shape) == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)

        keep_edge = data.edge_index[0] <= data.edge_index[1]
        edge_index = data.edge_index[:, keep_edge]
        edge_attr = data.edge_attr[keep_edge, :] if data.edge_attr is not None else data.edge_attr

        subgraphs = []

        for i in range(edge_index.size(1)):
            subgraph_edge_index = torch.hstack([edge_index[:, :i], edge_index[:, i + 1:]])
            subgraph_edge_attr = torch.vstack([edge_attr[:i], edge_attr[i + 1:]]) \
                if data.edge_attr is not None else data.edge_attr

            if data.edge_attr is not None:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                        num_nodes=data.num_nodes)
            else:
                subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                    num_nodes=data.num_nodes)

            subgraphs.append(
                Data(
                    x=data.x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        if len(subgraphs) == 0:
            subgraphs = [
                Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                     subgraph_idx=torch.tensor(0), subgraph_node_idx=torch.arange(data.num_nodes),
                     num_nodes=data.num_nodes,
                     )
            ]
        return subgraphs


class NodeDeleted(Graph2Subgraph):

    def to_subgraphs(self, data):

        subgraphs = []
        all_nodes = torch.arange(data.num_nodes)

        for i in range(data.num_nodes):
            subset = torch.cat([all_nodes[:i], all_nodes[i + 1:]])
            subgraph_edge_index, subgraph_edge_attr = subgraph(subset, data.edge_index, data.edge_attr,
                                                               relabel_nodes=False, num_nodes=data.num_nodes)

            subgraphs.append(
                Data(
                    x=data.x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs


class EgoNets(Graph2Subgraph):
    def __init__(self, num_hops, add_node_idx=False, process_subgraphs=lambda x: x, pbar=None):
        super().__init__(process_subgraphs, pbar)
        self.num_hops = num_hops
        self.add_node_idx = add_node_idx

    def to_subgraphs(self, data):

        subgraphs = []

        for i in range(data.num_nodes):

            _, _, _, edge_mask = k_hop_subgraph(i, self.num_hops, data.edge_index, relabel_nodes=False,
                                                num_nodes=data.num_nodes)
            subgraph_edge_index = data.edge_index[:, edge_mask]
            subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else data.edge_attr

            x = data.x
            if self.add_node_idx:
                # prepend a feature [0, 1] for all non-central nodes
                # a feature [1, 0] for the central node
                ids = torch.arange(2).repeat(data.num_nodes, 1)
                ids[i] = torch.tensor([ids[i, 1], ids[i, 0]])

                x = torch.hstack([ids, data.x]) if data.x is not None else ids.to(torch.float)

            subgraphs.append(
                Data(
                    x=x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs


class Explanation(Graph2Subgraph):
    def __init__(self, my, process_subgraphs=lambda x: x, pbar=None):
        super().__init__(process_subgraphs, pbar)
        self.my = my
    
    def to_subgraphs(self, data):
        data.to(self.my.device)
        subgraphs = [] 
        graph = data.edge_index
        with torch.no_grad():
            embeds = self.my.model.embedding(data)
        input_expl = self.my._create_explainer_input(graph, embeds)
        sampling_weights = self.my.explainer(input_expl).squeeze()
      
        k_min = (graph.size(0) * 5) // 100
        k_max = (graph.size(0) * 50) // 100
        
        for i, val in enumerate(range(k_min, k_max, 2)):
            hm = self.my._sample_k_graph(graph, sampling_weights, training=self.my.noise, size=val)

            subgraph_edge_index = graph[:, hm.long()]

            subgraphs.append(
                Data(
                    x=data.x, edge_index=subgraph_edge_index, subgraph_idx=torch.tensor(i),
                    subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                ).cpu()
            )
        data.cpu()
        return subgraphs
        

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def S2V_to_PyG(data):
    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'num_nodes', data.node_features.shape[0])
    setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())

    return new_data


def load_data(dataset, degree_as_tag, folder):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s.txt' % (folder, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    return [S2V_to_PyG(datum) for datum in g_list]


class PTCDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            name,
            transform=None,
            pre_transform=None,
    ):
        self.name = name
        self.url = 'https://github.com/weihua916/powerful-gnns/raw/master/dataset.zip'

        super(PTCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, self.name, name)

    @property
    def num_tasks(self):
        return 1  # it is always binary classification for the datasets we consider

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'

    @property
    def raw_file_names(self):
        return ['PTC.mat', 'PTC.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        folder = osp.join(self.root, self.name)
        path = download_url(self.url, folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)

        shutil.move(osp.join(folder, f'dataset/{self.name}'), osp.join(folder, self.name))
        shutil.rmtree(osp.join(folder, 'dataset'))

        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_data('PTC', degree_as_tag=False, folder=self.raw_dir)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def separate_data(self, seed, fold_idx):
        # code taken from GIN and adapted
        # since we only consider train and valid, use valid as test

        assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        labels = self.data.y.numpy()
        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        train_idx, test_idx = idx_list[fold_idx]

        return {'train': torch.tensor(train_idx), 'valid': torch.tensor(test_idx), 'test': torch.tensor(test_idx)}


def policy2transform(policy: str, num_hops, process_subgraphs=lambda x: x, pbar=None, dataset_name = None, device='cpu'):

    if policy == "edge_deleted":
        return EdgeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "node_deleted":
        return NodeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets":
        return EgoNets(num_hops, process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets_plus":
        return EgoNets(num_hops, add_node_idx=True, process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "original":
        return process_subgraphs
    raise ValueError("Invalid subgraph policy type")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='Data downloading and preprocessing')
    parser.add_argument('--dataset', type=str, default='ogbg-molhiv',
                        help='which dataset to preprocess (default: ogbg-molhiv)')
    parser.add_argument('--policies', type=str, nargs='+', help='which policies to preprocess (default: all)')
    args = parser.parse_args()

    policies = args.policies
    if policies is None:
        policies = ["edge_deleted", "node_deleted", "ego_nets", "ego_nets_plus", "original"]

    num_graphs = {
        'ogbg-molhiv': 41127,
        'ogbg-moltox21': 7831,
        'NCI1': 4110,
        'NCI109': 4127,
        'MUTAG': 188,
        'MUTAG_GT': 4337,
        'PROTEINS': 1113,
        'PTC': 344,
        'IMDB-BINARY': 1000,
        'IMDB-MULTI': 1500,
        'REDDIT-BINARY': 2000,
        'ZINC': [1000, 1000],
        'CSL': 150,
        'CEXP': 1200,
        'EXP': 1200,
    }
    process = lambda x: x
    if 'IMDB' in args.dataset or 'REDDIT' in args.dataset:
        process = OneHotDegree(135) if 'IMDB' in args.dataset else Constant()
    elif 'CSL' in args.dataset:
        process = OneHotDegree(5)

    num_hops = 2
    for policy in policies:

        if 'ogbg' in args.dataset:
            print()
        #    DatasetName = PygGraphPropPredDataset
        elif args.dataset == 'PTC':
            DatasetName = PTCDataset
        elif args.dataset == 'CSL':
            DatasetName = MyGNNBenchmarkDataset
        elif args.dataset in ['EXP', 'CEXP']:
            DatasetName = PlanarSATPairsDataset
        elif args.dataset == 'MUTAG_GT':
            DatasetName = MutagGTDataset
        else:
            DatasetName = TUDataset

        DatasetName(root="dataset/" + policy,
                              name=args.dataset,
                              pre_transform=policy2transform(policy=policy, num_hops=num_hops,
                                                             process_subgraphs=process,
                                                             pbar=iter(tqdm.tqdm(range(num_graphs[args.dataset]))),
                                                             dataset_name=args.dataset,
                                                             device=device
                                                             )
                              )

if __name__ == '__main__':
    main()
