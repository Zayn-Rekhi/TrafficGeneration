import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
from torch_geometric.utils import add_self_loops, degree



class NaiveMsgPass(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3: Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = (deg_inv_sqrt[row] + deg_inv_sqrt[col]).pow(-1.0)

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_j has shape [E, out_channels]
        tmp = torch.cat([x_i, x_j], dim=1)
        tmp = self.lin(tmp)

        # Step 4: Normalize node features.
        return tmp



class BlockGenerator(torch.nn.Module):
    def __init__(self, opt, device):    
        super(BlockGenerator, self).__init__()
        self.device = device
        self.latent_dim = opt['latent_dim']
        self.latent_ch = opt['n_ft_dim']
        

        if opt['aggr'] == 'Mean':
            self.global_pool = torch_geometric.nn.global_mean_pool
        elif opt['aggr'] == 'Max':
            self.global_pool = torch_geometric.nn.global_max_pool
        elif opt['aggr'] == 'Add':
            self.global_pool = torch_geometric.nn.global_add_pool
        elif opt['aggr'] == 'global_sort_pool':
            self.global_pool = torch_geometric.nn.global_sort_pool     
        elif opt['aggr'] == 'GlobalAttention':
            self.global_pool = torch_geometric.nn.GlobalAttention
        elif opt['aggr'] == 'Set2Set':
            self.global_pool = torch_geometric.nn.Set2Set
        elif opt['aggr'] == 'GraphMultisetTransformer':
            self.global_pool = torch_geometric.nn.GraphMultisetTransformer


        if opt['convlayer'] == 'GCNConv':
            self.convlayer = torch_geometric.nn.GCNConv
        elif opt['convlayer'] == 'NaiveMsgPass':
            self.convlayer = NaiveMsgPass
        else:
            self.convlayer = torch_geometric.nn.GCNConv

        
        self.pos_init = nn.Linear(2, int(self.latent_ch / 2))
        self.size_init = nn.Linear(2, int(self.latent_ch / 2))
        self.vel_init = nn.Linear(1, int(self.latent_ch / 2))
        
        self.type_init = nn.Linear(6, int(self.latent_ch / 2))
        self.direction_init = nn.Linear(2, int(self.latent_ch / 2))
        self.lane_index_init = nn.Linear(10, int(self.latent_ch / 2))

        self.e_conv1 = self.convlayer(int(self.latent_ch * 3), self.latent_ch)
        self.e_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.e_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        self.d_conv1 = self.convlayer(self.latent_ch, self.latent_ch)
        self.d_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.d_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        self.d_ft_init = nn.Linear(self.latent_dim, self.latent_ch * 6)

        # self.aggregate = nn.Linear(self.latent_ch * 6, self.latent_dim)
        self.aggregate = self.convlayer(self.latent_ch, self.latent_ch)

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        # Positioning
        self.d_posx_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_posx_1 = nn.Linear(self.latent_ch, 1)

        self.d_posy_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_posy_1 = nn.Linear(self.latent_ch, 1)

        # Size
        self.d_sizex_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_sizex_1 = nn.Linear(self.latent_ch, 1)

        self.d_sizey_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_sizey_1 = nn.Linear(self.latent_ch, 1)

        # Velocity/Direction
        self.d_velx_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_velx_1 = nn.Linear(self.latent_ch, 1)

        self.d_vely_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_vely_1 = nn.Linear(self.latent_ch, 1)

        self.d_act_type_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_act_type_1 = nn.Linear(self.latent_ch, 6)

        self.d_direction_cos_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_direction_cos_1 = nn.Linear(self.latent_ch, 1)

        self.d_direction_sin_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_direction_sin_1 = nn.Linear(self.latent_ch, 1)

        self.d_lane_index_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_lane_index_1 = nn.Linear(self.latent_ch, 10)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z
    
    def freeze_encoder(self):
        self.pos_init.weight.requires_grad = False
        self.pos_init.bias.requires_grad = False

        self.size_init.weight.requires_grad = False
        self.size_init.bias.requires_grad = False

        self.vel_init.weight.requires_grad = False
        self.vel_init.bias.requires_grad = False

        self.type_init.weight.requires_grad = False
        self.type_init.bias.requires_grad = False

        self.direction_init.weight.requires_grad = False
        self.direction_init.bias.requires_grad = False

        self.lane_index_init.weight.requires_grad = False
        self.lane_index_init.bias.requires_grad = False

        for param in self.e_conv1.parameters():
            param.requires_grad = False

        for param in self.e_conv2.parameters():
            param.requires_grad = False

        for param in self.e_conv3.parameters():
            param.requires_grad = False

    def encode(self, data):
        pos_org, size_org, vel_org, actor_type, lane_index, direction, edge_index = data.pos, data.dimen, data.vel, data.actor_type, data.lane_index, data.direction, data.edge_index

        direction = F.normalize(direction, dim=-1)

        pos = F.relu(self.pos_init(pos_org))
        size = F.relu(self.size_init(size_org))
        vel = F.relu(self.vel_init(vel_org))

        act_type = F.relu(self.type_init(actor_type))
        lane_idx = F.relu(self.lane_index_init(lane_index))
        direc = F.relu(self.direction_init(direction))

        # n_embd_0 = torch.unsqueeze(pos, dim =0)
        # filler = torch.zeros((pos.shape[0], pos.shape[1] * 3)).to(self.device) # TODO: Replace Later
        n_embd_0 = torch.cat((pos, size, vel, act_type, lane_idx, direc), 1)       

        n_embd_1 = F.relu(self.e_conv1(n_embd_0, edge_index))
        n_embd_2 = F.relu(self.e_conv2(n_embd_1, edge_index))
        n_embd_3 = F.relu(self.e_conv3(n_embd_2, edge_index))
        
        g_embd_0 = self.global_pool(n_embd_0, data.batch)
        g_embd_1 = self.global_pool(n_embd_1, data.batch)
        g_embd_2 = self.global_pool(n_embd_2, data.batch)
        g_embd_3 = self.global_pool(n_embd_3, data.batch)

        g_embd = torch.cat((g_embd_0, g_embd_1, g_embd_2, g_embd_3), 1)

        latent = self.aggregate(g_embd)


        return [mu, log_var]



    def decode(self, z, edge_index, condition=None):


        if isinstance(condition, torch.Tensor):
            z = torch.cat((z, condition), 1)

        z = self.d_ft_init(z).view(z.shape[0] * 6, -1)

        d_embd_0 = F.relu(z)
        d_embd_1 = F.relu(self.d_conv1(d_embd_0, edge_index))
        d_embd_2 = F.relu(self.d_conv2(d_embd_1, edge_index))
        d_embd_3 = F.relu(self.d_conv3(d_embd_2, edge_index))
            
        posx = F.relu(self.d_posx_0(d_embd_3))
        posx = self.d_posx_1(posx)

        posy = F.relu(self.d_posy_0(d_embd_3))
        posy = self.d_posy_1(posy)

        sizex = F.relu(self.d_sizex_0(d_embd_3))
        sizex = self.d_sizex_1(sizex)

        sizey = F.relu(self.d_sizey_0(d_embd_3))
        sizey = self.d_sizey_1(sizey)

        vel = F.relu(self.d_vel_0(d_embd_3))
        vel = self.d_vel_1(vel)

        acttype = F.relu(self.d_act_type_0(d_embd_3))
        acttype = self.d_act_type_1(acttype)

        direc_cos = F.relu(self.d_direction_cos_0(d_embd_3))
        direc_cos = self.d_direction_cos_1(direc_cos)

        direc_sin = F.relu(self.d_direction_sin_0(d_embd_3))
        direc_sin = self.d_direction_sin_1(direc_sin)

        laneidx = F.relu(self.d_lane_index_0(d_embd_3))
        laneidx = self.d_lane_index_1(laneidx)

        return posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx


    def forward(self, data):
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx = self.decode(z, data.edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        direc = torch.cat((direc_cos, direc_sin), 1)
        direc = F.normalize(direc, dim=-1)

        return pos, size, vel, acttype, direc, laneidx, log_var, mu
    
    def decoder_only(self, latent, edge_index):
        posx, posy, sizex, sizey, velx, vely, acttype, direc, laneidx = self.decode(latent, edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        vel = torch.cat((velx, vely), 1)

        return pos, size, vel, acttype, direc, laneidx



class AttentionBlockGenerator(BlockGenerator):
    def __init__(self, opt, device):
        super(AttentionBlockGenerator, self).__init__(opt, device)

        self.head = opt['head_num']

        use_head = ['GATConv', 'TransformerConv', 'GPSConv', 'GATv2Conv', 'SuperGATConv']

        if opt['convlayer'] == 'ChebConv':
            self.convlayer = torch_geometric.nn.ChebConv            
        elif opt['convlayer'] == 'SAGEConv':
            self.convlayer = torch_geometric.nn.SAGEConv
        elif opt['convlayer'] == 'GraphConv':
            self.convlayer = torch_geometric.nn.GraphConv
        elif opt['convlayer'] == 'GravNetConv':
            self.convlayer = torch_geometric.nn.GravNetConv
        elif opt['convlayer'] == 'GatedGraphConv':
            self.convlayer = torch_geometric.nn.GatedGraphConv
        elif opt['convlayer'] == 'ResGatedGraphConv':
            self.convlayer = torch_geometric.nn.ResGatedGraphConv     
        elif opt['convlayer'] == 'GATConv':
            self.convlayer = torch_geometric.nn.GATConv
        elif opt['convlayer'] == 'GATv2Conv':
            self.convlayer = torch_geometric.nn.GATv2Conv
        elif opt['convlayer'] == 'TransformerConv':
            self.convlayer = torch_geometric.nn.TransformerConv
        elif opt['convlayer'] == 'GPSConv':
            self.convlayer = torch_geometric.nn.GPSConv
        elif opt['convlayer'] == 'SuperGATConv':
            self.convlayer = torch_geometric.nn.SuperGATConv
            

        if opt['convlayer'] in use_head:
            self.d_conv1 = self.convlayer((-1, self.latent_ch), self.latent_ch, heads = self.head)
        else:
            self.d_conv1 = self.convlayer((-1, self.latent_ch), self.latent_ch)
        

        if opt['convlayer'] in use_head:
            self.e_conv1 = self.convlayer(int(self.latent_ch * 3), self.latent_ch, heads = self.head)
            self.e_conv2 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            self.e_conv3 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            self.d_conv2 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            self.d_conv3 = self.convlayer(self.latent_ch * self.head, self.latent_ch, heads = self.head)
            
            self.aggregate = nn.Linear(self.latent_ch * (self.head * 3 + 3), self.latent_dim)
        else:
            self.e_conv1 = self.convlayer(int(self.latent_ch * 3), self.latent_ch)
            self.e_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
            self.e_conv3 = self.convlayer(self.latent_ch, self.latent_ch)
            self.d_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
            self.d_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

            self.aggregate = nn.Linear(int(self.latent_ch * 6), self.latent_dim)        



        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)


        if opt['convlayer'] in use_head:
            self.d_posx_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_posx_1 = nn.Linear(self.latent_ch, 1)

            self.d_posy_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_posy_1 = nn.Linear(self.latent_ch, 1)

            self.d_sizex_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_sizex_1 = nn.Linear(self.latent_ch, 1)

            self.d_sizey_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_sizey_1 = nn.Linear(self.latent_ch, 1)

            self.d_vel_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_vel_1 = nn.Linear(self.latent_ch, 1)

            self.d_act_type_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_act_type_1 = nn.Linear(self.latent_ch, 6)

            self.d_direction_cos_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_direction_cos_1 = nn.Linear(self.latent_ch, 1)

            self.d_direction_sin_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_direction_sin_1 = nn.Linear(self.latent_ch, 1)

            self.d_lane_index_0 = nn.Linear(self.latent_ch * self.head, self.latent_ch)
            self.d_lane_index_1 = nn.Linear(self.latent_ch, 10)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))


class AttentionBlockGeneratorWithEmbeddings(AttentionBlockGenerator):
    def __init__(self, opt, device):    
        super().__init__(opt, device)

        self.text_encoder = nn.Linear(opt['embed_size'], opt['latent_dim'])

        self.d_ft_init = nn.Linear(self.latent_dim * 2, self.latent_ch * 6)

    
    def encode_text(self, x):
        x = self.text_encoder(x)
        return x
    
    def forward(self, data):
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        condition = self.encode_text(data.embeddings)
        posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx = self.decode(z, data.edge_index, condition)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        direc = torch.cat((direc_cos, direc_sin), 1)

        return pos, size, vel, acttype, direc, laneidx, log_var, mu
    

    def decoder_only(self, latent, edge_index, embed):
        condition = self.encode_text(embed)
        posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx = self.decode(latent, edge_index, condition)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        direc = torch.cat((direc_cos, direc_sin), 1)
        direc = F.normalize(direc, dim=-1)

        return pos, size, vel, acttype, direc, laneidx
