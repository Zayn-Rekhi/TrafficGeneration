import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from sentence_transformers import SentenceTransformer

import networkx as nx
import numpy as np
import os
import json
from tqdm import tqdm

from utils import graph2vector_processed


class CustomData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'lane_index' or key == 'direction':
            return 0  # concatenate along node dimension
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_index' or key == 'direction':
            # no offset needed — lane_index values are one-hot vectors
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class TrafficDataset(Dataset):
    """
    Construct a graph given a json file containing the traffic scenario configurations.
    """

    map_actor_types = {
        "Not An Actor (N/A)": 0,
        "ego_vehicle": 1,
        "car": 2,
        "truck": 2,
        "van": 2,
        "bus": 3,
        "tram": 3,
        "pedestrian": 4,
        "person_sitting": 4,
        "bicycle": 5,
        "cyclist": 5,
    }

    map_direction = {
        "-x": 0,
        "+x": 1,
    }

    map_lane_index = {str(x): x for x in range(0, 10)}

    def __init__(self, root, transform=None, pre_transform=None) -> None:
        """
        :param data_root: The root directory of the data.
        :param generator: The generator to be used to generate the data.
        :param fp_pg: The file path to the probabilistic grammar.
        """
        super().__init__(root, transform, pre_transform)

        assert os.path.exists(root), f"Data root {root} does not exist."
        self.data_root = root
        self.data, self.embeddings = self.load_data()
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int):
        return self.construct_graph_from_data(self.data[idx])
        # graph, features, mask, adjacency matrix

    def load_data(self):
        ls_folders = os.listdir(self.data_root)
        all_data, all_embeddings = [], []

        for folder in ls_folders:
            data, embeds = self._load_data(os.path.join(self.data_root, folder))
            all_data.extend(data)
            all_embeddings.extend(embeds)
            print("TrafficDataset.load_data: Loaded data from folder", folder)
            
        return all_data, all_embeddings

    def _load_data(self, path):
        ls_fn = os.listdir(path)
        ls_fn = [fn for fn in ls_fn if fn.endswith(".json")]  # Only load json files
        ls_fn.sort()

        all_data = []
        all_embeddings = []
        for fn in tqdm(ls_fn, desc="Loading data..."):
            with open(os.path.join(path, fn)) as f:
                data = json.load(f)

            all_data.append(data)
            all_embeddings.append(data['description'])

        all_embeddings = self.embedder(all_embeddings)

        print(f"TrafficDataset.load_data: Number of data samples: {len(all_data)}")
        print(
            f"TrafficDataset.load_data: Number of data samples discarded: {len(ls_fn) - len(all_data)}"
        )
        return all_data, all_embeddings

    def construct_graph_from_data(self, data):
        """
        Transform the data to a probabilistic grammar.
        :param data: The input data, a dictionary.
        """

        graph = nx.Graph()
        actors = data["actors"]
        
        for actor in actors:
            curr_idx = len(graph.nodes)
            graph.add_node(curr_idx, 
                           posx=actor["location_imu_x"], 
                           posy=actor["location_imu_y"],
                           width=actor["dimensions_width"],
                           length=actor["dimensions_length"],
                           velx=actor["direction_imu_x"],
                           vely=actor["direction_imu_y"],
                           direction=actor["yaw"],
                           type=self._encode_type(actor["type"].lower()),
                           lane_index=self._encode_lane_index(actor["lane_index"]))
            
        for idx in range(len(graph.nodes)):
            for idx2 in range(idx + 1, len(graph.nodes)):
                graph.add_edge(idx, idx2)

        assert graph.number_of_nodes() == 6, f"graph does not contain 6 nodes: {graph.number_of_nodes()}, IDX: {data}"
        assert graph.number_of_edges() == 15, f"graph does not contain 15 edges: {graph.number_of_edges()}"

        node_pos, node_size, node_vel, actor_type, lane_index, direction, edge_list, node_idx = graph2vector_processed(graph)       
        
        return CustomData(edge_index=torch.tensor(edge_list, dtype=torch.int64), 
                          pos=torch.tensor(node_pos, dtype=torch.float32),
                          dimen=torch.tensor(node_size, dtype=torch.float32),
                          vel=torch.tensor(node_vel, dtype=torch.float32),
                          actor_type=torch.tensor(actor_type, dtype=torch.float32),
                          lane_index=torch.tensor(lane_index, dtype=torch.float32),
                          direction=torch.tensor(direction, dtype=torch.float32),
                          node_idx=torch.tensor(node_idx, dtype=torch.float32))
    

    def _encode_type(self, x):
        one_hot = np.zeros(1 + max(self.map_actor_types.values()))
        one_hot[self.map_actor_types.get(x, 0)] = 1
        return one_hot
        
    
    def _encode_lane_index(self, x):
        one_hot = np.zeros(1 + max(self.map_lane_index.values()))
        one_hot[self.map_lane_index.get(x, 0)] = 1
        return one_hot

    @staticmethod
    def translate_dir(input_dir, output_dir):
        assert os.path.exists(input_dir)
        assert os.path.exists(output_dir)

        ls_all_json = os.listdir(input_dir)
        ls_all_json = [f for f in ls_all_json if f.endswith(".json")]
        ls_all_json.sort()

        for json_file in tqdm(
            ls_all_json, desc="Translating Graphs and saving to JSON..."
        ):
            input_json_path = os.path.join(input_dir, json_file)
            output_json_path = os.path.join(output_dir, json_file)

            with open(input_json_path, "r") as file:
                input_data = json.load(file)

            transformed_data = TrafficDataset.translate_graph_to_data(input_data)

            with open(output_json_path, "w") as file:
                json.dump(transformed_data, file, indent=2)

            # print("JSON transformation complete. Check output at:", output_json_path)


if __name__ == "__main__":

    data_root = "/home/zayn/dataset_processed_traffic/inD-dataset-v1.1"
    dataset = TrafficDataset(data_root)
    for idx, data in tqdm(
        enumerate(dataset), desc="Sanity check by iterating over the whole dataset"
    ):
        print(idx)
        break
    #     # print(graph)
    #     # print(features)
    #     # print(mask)
    #     # print(adjacency_matrix)

    loader = DataLoader(
        dataset, batch_size=1, num_workers=1
    )

    for i, data in enumerate(loader):
        print(i)
        print(data.edge_index)
        print(data.edge_index.shape)        

    # # ==========================================
    # ## Demo for translating the graphs
    # # ==========================================
    # input_json_dir = "../logs/KITTI/MetaSimVAE_Jan05-15-59-22/datagen/"
    # output_json_dir = "../logs/KITTI/MetaSimVAE_Jan05-15-59-22/translated/"
    # TrafficDataset.translate_dir(input_json_dir, output_json_dir)
