import networkx as nx
import numpy as np
import os
import yaml
import io
from PIL import Image

import matplotlib.pyplot as plt


def read_train_yaml(checkpoint_name, filename = "train.yaml"):
    with open(os.path.join(checkpoint_name, filename), "rb") as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return opt


def get_node_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_node_attributes(g, keys).items())
    attri = np.array(attri)   
    attri = attri[:,1]
    attri = np.array(attri, dtype = dtype)
    return attri

def get_node_attribute_cat(g, keys, dtype, default = None):
    attri = list(nx.get_node_attributes(g, keys).values())
    attri = np.stack(attri, dtype=dtype, axis=0)   
    return attri

def graph2vector_processed(g):
    num_nodes = g.number_of_nodes()

    posx = get_node_attribute(g, 'posx', np.double)
    posy = get_node_attribute(g, 'posy', np.double)

    actor_type = get_node_attribute_cat(g, 'type', np.double)
    lane_index = get_node_attribute_cat(g, 'lane_index', np.double)
    direction = get_node_attribute_cat(g, 'direction', np.double)


    edge_list = np.array(list(g.edges()), dtype=np.int_)
    edge_list = np.transpose(edge_list)

    node_pos = np.stack((posx, posy), 1)

    node_idx = np.stack((np.arange(num_nodes) / num_nodes, np.arange(num_nodes) / num_nodes), axis = 1)
    
    return node_pos, actor_type, lane_index, direction, edge_list, node_idx


def plot_comparison(target, output, edge_index, config, save_dir=None):
    target = target.detach().cpu().numpy()
    output = output.detach().cpu().numpy()

    target[:, 0] = (target[:, 0] * config['x'][1]) + config['x'][0]
    target[:, 1] = (target[:, 1] * config['y'][1]) + config['y'][0]

    output[:, 0] = (output[:, 0] * config['x'][1]) + config['x'][0]
    output[:, 1] = (output[:, 1] * config['y'][1]) + config['y'][0]

    edge_index = edge_index.detach().cpu().numpy()

    plt.figure(figsize=(10, 5))

    ax1 = plt.subplot(1, 2, 1)
    for i, j in edge_index.T:
        x = [target[i, 0], target[j, 0]]
        y = [target[i, 1], target[j, 1]]
        ax1.plot(x, y, color='lightgray', linewidth=0.5, zorder=1)

    ax1.scatter(target[:, 0], target[:, 1], color='royalblue', label='Target', edgecolors='k', zorder=2)
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = plt.subplot(1, 2, 2)
    for i, j in edge_index.T:
        x = [output[i, 0], output[j, 0]]
        y = [output[i, 1], output[j, 1]]
        ax2.plot(x, y, color='lightgray', linewidth=0.5, zorder=1)

    ax2.scatter(output[:, 0], output[:, 1], color='crimson', label='Output', edgecolors='k', zorder=2)
    ax2.set_title('Model Prediction')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_dir:
        print("Saved: ", save_dir)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=300)
        plt.close()

    fig = plt.gcf()
    return fig2img(fig)

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png') # Save the figure to the buffer as a PNG
    buf.seek(0) # Rewind the buffer to the beginning
    img = Image.open(buf) # Open the buffer as a PIL Image
    return img