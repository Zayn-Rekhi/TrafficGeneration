import networkx as nx
import numpy as np
import os
import yaml
import io
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg

import textwrap

import matplotlib as mpl
import torch

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

    width = get_node_attribute(g, 'width', np.double)
    length = get_node_attribute(g, 'length', np.double)

    node_vel = get_node_attribute(g, 'vel', np.double)

    yaw_sin = get_node_attribute(g, 'yaw_sin', np.double)
    yaw_cos = get_node_attribute(g, 'yaw_cos', np.double)

    actor_type = get_node_attribute_cat(g, 'type', np.double)
    lane_index = get_node_attribute_cat(g, 'lane_index', np.double)

    edge_list = np.array(list(g.edges()), dtype=np.int_)
    edge_list = np.transpose(edge_list)
    

    node_vel = np.expand_dims(node_vel, axis=1)

    node_pos = np.stack((posx, posy), 1)
    node_size = np.stack((width, length), 1)
    direction = np.stack((yaw_cos, yaw_sin), 1)


    node_idx = np.stack((np.arange(num_nodes) / num_nodes, np.arange(num_nodes) / num_nodes), axis = 1)
    
    return node_pos, node_size, node_vel, actor_type, lane_index, direction, edge_list, node_idx


def get_class_value(x):
    return np.argmax(x, axis=1)


def denormalize(arr, config, key1_dim, key2_dim=None):
    arr[:, 0] = (arr[:, 0] * config[key1_dim][1]) + config[key1_dim][0]
    
    if key2_dim:
        arr[:, 1] = (arr[:, 1] * config[key2_dim][1]) + config[key2_dim][0]
    
    return arr


def draw(ax, pos, size, vel, edge_index, direction, actor_type, lane_index, title, cmap, path, location_id, px_to_utm, config):
    image = mpimg.imread(path[0])

    ax.imshow(image, zorder=0)

    px_to_utm = float(px_to_utm[0])
    loc_id = int(location_id[0])

    ax.set_xlim(int(config[loc_id]["x_lim"][0] / config["scale_down_factor"]),
                int(config[loc_id]["x_lim"][1] / config["scale_down_factor"]))

    ax.set_ylim(int(config[loc_id]["y_lim"][0] / config["scale_down_factor"]),
                int(config[loc_id]["y_lim"][1] / config["scale_down_factor"]))


    # for i, j in edge_index.T:
    #     x = [pos[i, 0], pos[j, 0]]
    #     y = [pos[i, 1], pos[j, 1]]
    #     ax.plot(x, y, color='lightgray', linewidth=0.5, zorder=1)

    # Marker shapes for up to 5 actor types
    marker_map = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'P'}

    for i in range(len(pos)):
        color = cmap(lane_index[i] % 10 / 10)
        center_x, center_y = pos[i]
        width, length = size[i]

        width = (width / px_to_utm) / config["scale_down_factor"]
        length = (length / px_to_utm) / config["scale_down_factor"]

        print(center_x, center_y, px_to_utm, config["scale_down_factor"])

        center_x = (center_x / px_to_utm) / config["scale_down_factor"]
        center_y = (-center_y / px_to_utm) / config["scale_down_factor"]
        
        print(center_x, center_y)
        marker = marker_map.get(actor_type[i], 'x')

        direction_x, direction_y = direction[i]
        cur_vel = vel[i][0]
        yaw = np.degrees(np.arctan2(-direction_y, direction_x)) + config[int(location_id)]["deg_offset"]

        anchor = (center_x - length // 2,
                  center_y - width // 2)

        print(anchor, center_x, center_y, length, width )


        if width == 0 and length == 0:
            ax.scatter(center_x, center_y, marker=marker, color=color, edgecolors='k', s=40, zorder=4)
        else:
            rect = Rectangle(anchor, length, width,
                            linewidth=1, edgecolor='k', facecolor='blue', zorder=3)
            t2 = mpl.transforms.Affine2D().rotate_deg_around(center_x, center_y, yaw) + ax.transData
            rect.set_transform(t2)
            ax.add_patch(rect)

        # Draw velocity arrow
        ax.arrow(center_x, center_y, direction_x * cur_vel, -direction_y * cur_vel,
                   head_width=2, head_length=2, fc="red", ec="red", clip_on=False, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("X Axis (Front) - IMU")
    ax.set_ylabel("Y Axis (Left) - IMU")

    # ax.set_xlim(0, image.shape[1])
    # ax.set_ylim(image.shape[0], 0)
    
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_comparison(target, output, edge_index, config, opts, idx):
    save_dir = os.path.join(opts['log_dir'], f"vis/plot_{idx}.jpg")

    orig_pos, orig_size, orig_vel, orig_acttype, orig_direc, orig_laneidx = (
        target.pos, target.dimen, target.vel, target.actor_type, target.direction, target.lane_index
    )

    target_attr = (orig_pos, orig_size, orig_vel, orig_acttype, orig_direc, orig_laneidx)
    orig_pos, orig_size, orig_vel, orig_acttype, orig_direc, orig_laneidx = parse_by_actors(target_attr)
    
    out_attr = (output[0], output[1], output[2], output[3], output[4], output[5])
    pos, size, vel, acttype, direc, laneidx = parse_by_actors(out_attr)

    orig_pos = orig_pos.detach().cpu().numpy()
    orig_size = orig_size.detach().cpu().numpy()
    orig_vel = orig_vel.detach().cpu().numpy()
    orig_direc = orig_direc.detach().cpu().numpy()

    pos = pos.detach().cpu().numpy()
    size = size.detach().cpu().numpy()
    vel = vel.detach().cpu().numpy()
    direc = direc.detach().cpu().numpy()
    
    orig_acttype = get_class_value(orig_acttype.detach().cpu().numpy())
    orig_laneidx = get_class_value(orig_laneidx.detach().cpu().numpy())
    acttype = get_class_value(acttype.detach().cpu().numpy())
    laneidx = get_class_value(laneidx.detach().cpu().numpy())

    # De-normalize values
    orig_pos = denormalize(orig_pos, config, 'location_imu_x', 'location_imu_y')
    pos = denormalize(pos, config, 'location_imu_x', 'location_imu_y')

    orig_size = denormalize(orig_size, config, 'dimensions_width', 'dimensions_length')
    size = denormalize(size, config, 'dimensions_width', 'dimensions_length')

    orig_vel = denormalize(orig_vel, config, 'vel_mag')
    vel = denormalize(vel, config, 'vel_mag')

    edge_index = edge_index.detach().cpu().numpy()
    
    cmap = get_cmap('tab10')

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    draw(ax1, orig_pos, orig_size, orig_vel, edge_index, orig_direc, orig_acttype, orig_laneidx, 'Ground Truth', cmap, target.path, target.location_id, target.px_to_utm, opts['uniD_config'])

    ax2 = plt.subplot(1, 2, 2)
    draw(ax2, pos, size, vel, edge_index, direc, acttype, laneidx, 'Model Prediction', cmap, target.path, target.location_id, target.px_to_utm, opts['uniD_config'])

    title = target.prompt[0]
    wrapped_title = "\n".join(textwrap.wrap(".".join(title), width=80))

    plt.suptitle(wrapped_title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])


    if save_dir:
        print("Saved: ", save_dir)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=300)
        plt.close()


def plot_output(output, latent, edge_index, target, config, opts, save_dir=None):
    latent = latent.detach().cpu().numpy()
    pos, size, vel, acttype, direc, laneidx = output[0], output[1], output[2], output[3], output[4], output[5]

    pos = pos.detach().cpu().numpy()
    size = size.detach().cpu().numpy()
    vel = vel.detach().cpu().numpy()
    direc = direc.detach().cpu().numpy()

    acttype = get_class_value(acttype.detach().cpu().numpy())
    laneidx = get_class_value(laneidx.detach().cpu().numpy())

    pos = denormalize(pos, config, 'location_imu_x', 'location_imu_y')
    size = denormalize(size, config, 'dimensions_width', 'dimensions_length')
    vel = denormalize(vel, config, 'vel_mag')


    edge_index = edge_index.detach().cpu().numpy()
    
    cmap = get_cmap('tab10')

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(latent, cmap=cmap)

    ax2 = plt.subplot(1, 2, 2)
    draw(ax2, pos, size, vel, edge_index, direc, acttype, laneidx, 'Model Prediction', cmap, target.path, target.location_id, target.px_to_utm, opts['uniD_config'])

    plt.tight_layout()

    if save_dir:
        print("Saved: ", save_dir)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=300)
        plt.close()

    fig = plt.gcf()
    return fig2img(fig)


def parse_by_actors(output):
    pos, size, vel, acttype, direc, laneidx = output
    mask = torch.argmax(acttype, axis=1) != 0
    return pos[mask, :], size[mask, :], vel[mask, :], acttype[mask, :], direc[mask, :], laneidx[mask, :]


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png') # Save the figure to the buffer as a PNG
    buf.seek(0) # Rewind the buffer to the beginning
    img = Image.open(buf) # Open the buffer as a PIL Image
    return img


def to_boxes(size, pos):
    return torch.cat((pos, size + pos), dim = -1)
