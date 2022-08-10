import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

""" 
The function in this file is largely copied from the orginal PGExplainer codebase. The decision was made to largely copy this file to ensure
that the graph visualization between the original and replicate results would be as similar as possible. Additional comments were added
to clarify the code. 
"""

def plot(graph, edge_weigths, idx, args=None, show=False, multiple=False, mask_idx=None):
    """
    Function that can plot an explanation (sub)graph and store the image.
    If multiple is True it stores all the subgraphs separately.

    :param graph: graph provided by explainer
    :param edge_weigths: Mask of edge weights provided by explainer
    :param idx: Node index of interesting node
    :param thresh_min: total number of edges
    :param args: Object containing arguments from configuration
    :param show: flag to show plot made
    :param multiple: flag to print separate subgraphs
    :mask_idx: index of the subgraph to name the file
    """
    # Set thresholds
    thres = 0.

    # Init edges
    pos_edges = []
    weights = []
    # Select all edges and nodes to plot
    for i in range(edge_weigths.shape[0]):
        # Select important edges
        if edge_weigths[i] > thres and not graph[0][i] == graph[1][i]:
            pos_edges.append((graph[0][i].item(),graph[1][i].item()))
            weights.append(edge_weigths[i])

    # Initialize graph object
    G = nx.Graph()
    
    #colors = ['orange','red','lime','green','blue','orchid','darksalmon','darkslategray','gold','bisque','tan','lightseagreen','indigo','navy']
    # Format edges
    edges = [(pair[0].item(), pair[1].item()) for pair in graph.T]
    # Obtain all unique nodes
    nodes = np.unique(graph.T)
    #print(nodes)

    # Add all unique nodes and all edges
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    # Let the graph generate all positions
    pos = nx.kamada_kawai_layout(G)

    pos_edges = [(u, v) for (u, v) in pos_edges if u in G.nodes() and v in G.nodes()]

    nx.draw_networkx_nodes(G,
                            pos,
                            nodelist=nodes,
                            node_color='red',
                            node_size=50)


    # Draw an edge
    nx.draw_networkx_edges(G,
                           pos,
                           width=3.5,
                           alpha=1,
                           edge_color='green',
                           style='dashed')

    # Draw all pos edges
    nx.draw_networkx_edges(G,
                           pos,
                           edgelist=pos_edges,
                           width=5,
                           alpha=weights)
    plt.axis('off')
    if show:
        plt.show()
    else:
        if not multiple:
            save_path = f'./qualitative/e_{args.explainer}/m_{args.model}/d_{args.dataset}/p_{args.subgraph_policy}/'
    
            # Generate folders if they do not exist
            Path(save_path).mkdir(parents=True, exist_ok=True)
    
            # Save figure
            plt.savefig(f'{save_path}{idx}.png')
            plt.clf()
        else:
            save_path = f'./qualitative/esan/m_{args.model}/d_{args.dataset}/p_{args.subgraph_policy}/i_{idx}/'
    
            # Generate folders if they do not exist
            Path(save_path).mkdir(parents=True, exist_ok=True)
    
            # Save figure
            plt.savefig(f'{save_path}{idx}_{mask_idx}.png')
            plt.clf()


def plot2(graph, edge_weights, idx, args=None, show=False):
    """
    Function that plots a graph with a color grade on edges and store the image.

    :param graph: graph provided by explainer
    :param edge_weights: Mask of edge weights provided by explainer
    :param idx: Node index of interesting node
    :param args: Object containing arguments from configuration
    :param show: flag to show plot made
    """

    # Init edges
    pos_edges = []
    weights = []
    # Select all edges and nodes to plot
    for i in range(edge_weights.shape[0]):
        # Select important edges
        if not graph[0][i] == graph[1][i]:
            pair = tuple(sorted((graph[0][i].item(),graph[1][i].item())))
            # Count (0,1) and (1,0) once
            if pair in pos_edges:
                j = pos_edges.index(pair)
                weights[j] += int(edge_weights[i].item())
            else:
                pos_edges.append(pair)
                weights.append(int(edge_weights[i].item()))
    # Rescale from [0,20] to [0,10]
    weights = [w/2 for w in weights]

    # Initialize graph object
    G = nx.Graph()
    
    # Obtain all unique nodes
    nodes = np.unique(graph.T)
    #print(nodes)

    # Add all unique nodes and all edges
    G.add_nodes_from(nodes)
    pos_edges = [(u, v) for (u, v) in pos_edges if u in G.nodes() and v in G.nodes()]
    G.add_edges_from(pos_edges)
    # Let the graph generate all positions
    pos = nx.kamada_kawai_layout(G)

    pos_edges = [(u, v) for (u, v) in pos_edges if u in G.nodes() and v in G.nodes()]

    nx.draw_networkx_nodes(G,
                            pos,
                            nodelist=nodes,
                            node_color='red',
                            node_size=50)

    # Draw all pos edges
    cmap = plt.cm.viridis_r
    bounds = [i for i in range(args.mask_num+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    nx.draw_networkx_edges(G,
                           pos,
                           edgelist=pos_edges,
                           width=5,
                           alpha=1,
                           edge_color=weights,
                           edge_vmin=0,
                           edge_vmax=args.mask_num,
                           edge_cmap=cmap)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),orientation='vertical',label='Frequency')
    plt.axis('off')
    if show:
        plt.show()
    else:
        save_path = f'./qualitative/esan/m_{args.model}/d_{args.dataset}/p_{args.subgraph_policy}/i_{idx}/'
    
        # Generate folders if they do not exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
    
        # Save figure
        plt.savefig(f'{save_path}color_{idx}.png')
        plt.clf()


def activation_hist(layer_activations, layer_name, args):
    fig, axs = plt.subplots(args.epochs, 1, sharey=True, figsize=(5,1), tight_layout=True)
    i = 0
    for e in layer_activations:
        axs[i].hist(e, bins=50)
        i += 1

    fig.set_figheight(30)
    fig.set_figwidth(5)
    fig.suptitle(f'{layer_name}')

    save_path = f'./qualitative/BatchNorm/m_{args.model}/d_{args.dataset}/'
    
    # Generate folders if they do not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(f'{save_path}histo_{layer_name}.png')
    plt.clf()