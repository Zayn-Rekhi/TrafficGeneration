import networkx as nx
import numpy as np
import os
import yaml
import io
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.cm import get_cmap


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

def get_class_value(x):
    return np.argmax(x, axis=1)


def plot_comparison(target, output, edge_index, config, save_dir=None):
    orig_pos, orig_acttype, orig_direction, orig_laneidx = target.pos, target.actor_type, target.direction, target.lane_index
    pos, acttype, direction, laneidx, _, _ = output

    orig_pos = orig_pos.detach().cpu().numpy()
    pos = pos.detach().cpu().numpy()
    
    orig_acttype = get_class_value(orig_acttype.detach().cpu().numpy())
    orig_direction = get_class_value(orig_direction.detach().cpu().numpy())
    orig_laneidx = get_class_value(orig_laneidx.detach().cpu().numpy())

    acttype = get_class_value(acttype.detach().cpu().numpy())
    direction = get_class_value(direction.detach().cpu().numpy())
    laneidx = get_class_value(laneidx.detach().cpu().numpy())

    # De-normalize positions
    orig_pos[:, 0] = (orig_pos[:, 0] * config['x'][1]) + config['x'][0]
    orig_pos[:, 1] = (orig_pos[:, 1] * config['y'][1]) + config['y'][0]
    pos[:, 0] = (pos[:, 0] * config['x'][1]) + config['x'][0]
    pos[:, 1] = (pos[:, 1] * config['y'][1]) + config['y'][0]

    edge_index = edge_index.detach().cpu().numpy()

    def draw(ax, pos, edge_index, actor_type, direction, lane_index, title, cmap):
        # Draw edges
        for i, j in edge_index.T:
            x = [pos[i, 0], pos[j, 0]]
            y = [pos[i, 1], pos[j, 1]]
            ax.plot(x, y, color='lightgray', linewidth=0.5, zorder=1)

        # Marker shapes for 5 actor types
        marker_map = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'P'}

        for i in range(len(pos)):
            marker = marker_map.get(actor_type[i], 'x')
            color = cmap(lane_index[i] % 10 / 10)  # normalize for tab10

            ax.scatter(pos[i, 0], pos[i, 1], marker=marker, color=color, edgecolors='k', s=50, zorder=2)
            ax.arrow(pos[i, 0], pos[i, 1], 3 if direction[i] == 1 else -3, 0, head_width=1, head_length=1)

            
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.5)

    cmap = get_cmap('tab10')  # For lane_index

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    draw(ax1, orig_pos, edge_index, orig_acttype, orig_direction, orig_laneidx, 'Ground Truth', cmap)

    ax2 = plt.subplot(1, 2, 2)
    draw(ax2, pos, edge_index, acttype, direction, laneidx, 'Model Prediction', cmap)

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