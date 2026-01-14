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



class SceneGenerator(torch.nn.Module):
    def __init__(self, opt, device):    
        super(SceneGenerator, self).__init__()
        self.device = device
        self.latent_dim = opt['latent_dim']
        self.latent_ch = opt['n_ft_dim']
        self.n_components = opt['n_components']
        self.n_actors = opt['n_actors']

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
        self.lane_index_init = nn.Linear(2, int(self.latent_ch / 2))

        self.e_conv1 = self.convlayer(int(self.latent_ch * 3), self.latent_ch)
        self.e_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.e_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        self.d_conv1 = self.convlayer(self.latent_ch, self.latent_ch)
        self.d_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
        self.d_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

        # self.d_ft_init = nn.Linear(self.latent_dim, self.latent_ch)
        self.d_ft_init = nn.Linear(self.latent_dim, self.latent_ch * 6)

        self.aggregate = nn.Linear(self.latent_ch * 7, self.latent_dim)
        # self.aggregate = self.convlayer(self.latent_ch, self.latent_ch)

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim * self.n_components)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim * self.n_components)
        self.fc_pi = nn.Linear(self.latent_dim, self.n_components)

        # --- Learned GMM PRIOR: p(y), p(z|y) ---
        self.prior_mu     = nn.Parameter(torch.zeros(self.n_components, self.latent_dim))      # [K,D]
        self.prior_logvar = nn.Parameter(torch.zeros(self.n_components, self.latent_dim))      # [K,D]
        self.prior_logits = nn.Parameter(torch.zeros(self.n_components))                       # [K]

        # Optional sensible init
        nn.init.normal_(self.prior_mu, std=0.1)
        self.prior_logvar.data.fill_(0.0)   # log(1)
        self.prior_logits.data.zero_()      # uniform prior after softmax

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

        self.d_vel_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_vel_1 = nn.Linear(self.latent_ch, 1)

        self.d_act_type_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_act_type_1 = nn.Linear(self.latent_ch, 6)

        self.d_direction_cos_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_direction_cos_1 = nn.Linear(self.latent_ch, 1)

        self.d_direction_sin_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_direction_sin_1 = nn.Linear(self.latent_ch, 1)

        self.d_lane_index_0 = nn.Linear(self.latent_ch, self.latent_ch)
        self.d_lane_index_1 = nn.Linear(self.latent_ch, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))


    def sample(self, mu, logvar, pi_logits, tau=0.5, hard=True):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """

        B, K, D = mu.size(0), self.n_components, self.latent_dim
        y = F.gumbel_softmax(pi_logits, tau=tau, hard=hard, dim=-1)  # <-- logits in
        std = torch.exp(0.5 * logvar.view(B, K, D)).clamp_min(1e-3)  # floor var
        eps = torch.randn(B, K, D, device=mu.device)
        z_k = mu.view(B, K, D) + std * eps                           # [B,K,D]
        z = torch.sum(z_k * y.unsqueeze(-1), dim=1)                  # [B,D]
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
        print(n_embd_1.shape)
        n_embd_2 = F.relu(self.e_conv2(n_embd_1, edge_index))
        print(n_embd_2.shape)
        n_embd_3 = F.relu(self.e_conv3(n_embd_2, edge_index))
        print(n_embd_3.shape)

        
        g_embd_0 = self.global_pool(n_embd_0, data.batch)
        g_embd_1 = self.global_pool(n_embd_1, data.batch)
        g_embd_2 = self.global_pool(n_embd_2, data.batch)
        g_embd_3 = self.global_pool(n_embd_3, data.batch)

        g_embd = torch.cat((g_embd_0, g_embd_1, g_embd_2, g_embd_3), 1)

        latent = self.aggregate(g_embd)

        mu = self.fc_mu(latent)
        log_var = self.fc_var(latent)
        pi_logits = self.fc_pi(latent)
        pi = F.softmax(pi_logits, dim = 1)

        return mu, log_var, pi, pi_logits



    def decode(self, z, edge_index, condition=None):


        if isinstance(condition, torch.Tensor):
            z = torch.cat((z, condition), 1)

        z = self.d_ft_init(z).view(z.shape[0] * self.n_actors, -1)
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
        mu, log_var, pi, pi_logits = self.encode(data)
        z = self.sample(mu, log_var, pi_logits)
        posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx = self.decode(z, data.edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        direc = torch.cat((direc_cos, direc_sin), 1)
        direc = F.normalize(direc, dim=-1)

        return pos, size, vel, acttype, direc, laneidx, log_var, mu, pi, pi_logits
    
    def decoder_only(self, latent, edge_index, batch_dec):
        posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx = self.decode(latent, edge_index)
        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        direc = torch.cat((direc_cos, direc_sin), 1)
        direc = F.normalize(direc, dim=-1)

        return pos, size, vel, acttype, direc, laneidx



class AttentionSceneGenerator(SceneGenerator):
    def __init__(self, opt, device):
        super(AttentionSceneGenerator, self).__init__(opt, device)

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
            self.e_conv1 = self.convlayer(int(self.latent_ch * 4), self.latent_ch)
            self.e_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
            self.e_conv3 = self.convlayer(self.latent_ch, self.latent_ch)
            self.d_conv2 = self.convlayer(self.latent_ch, self.latent_ch)
            self.d_conv3 = self.convlayer(self.latent_ch, self.latent_ch)

            self.aggregate = nn.Linear(int(self.latent_ch * 6), self.latent_dim)        



        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim * self.n_components)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim * self.n_components)
        self.fc_pi = nn.Linear(self.latent_dim, self.n_components)


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
            self.d_lane_index_1 = nn.Linear(self.latent_ch, 2)



        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))



class AttentionSceneGeneratorWithEmbeddings(AttentionSceneGenerator):
    def __init__(self, opt, device):    
        super().__init__(opt, device)

        self.d_ft_init = nn.Linear(self.latent_dim * 2, self.latent_ch * self.n_actors)
        self.embed_size = opt['embed_size']
        self.n_heads = opt.get('text_heads', 4)
        self.cross_attn_dim = self.latent_dim  # project text embeddings to latent_dim for cross-attention

        # Project text embeddings to cross-attention dimension
        self.text_proj = nn.Linear(self.embed_size, self.cross_attn_dim)
        
        # Multihead cross-attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.cross_attn_dim,
                                                num_heads=self.n_heads,
                                                batch_first=True)

        # Optional feedforward after cross-attention
        self.text_ffn = nn.Sequential(
            nn.Linear(self.cross_attn_dim, self.cross_attn_dim),
            nn.ReLU(),
            nn.Linear(self.cross_attn_dim, self.cross_attn_dim)
        )
    
    def encode_text(self, text_embeddings, scene_latent=None):
        """
        text_embeddings: [B, L, embed_size] (batch, seq_len, embedding_dim)
        scene_latent: [B, latent_dim] (optional conditioning query)
        """
        text_proj = self.text_proj(text_embeddings)  # [B, L, latent_dim]

        if scene_latent is None:
            text_out = text_proj.mean(dim=1)  # [B, latent_dim]
        else:
            query = scene_latent.unsqueeze(1)  # [B, 1, latent_dim]
            key = text_proj
            value = text_proj
            attn_out, _ = self.cross_attn(query=query, key=key, value=value)
            attn_out = attn_out.squeeze(1)  # [B, latent_dim]
            text_out = self.text_ffn(attn_out)  # optional FFN

        return text_out

    def forward(self, data):
        # Encode scene graph
        mu, log_var, pi, pi_logits = self.encode(data)
        z = self.sample(mu, log_var, pi_logits)

        # Encode text with cross-attention
        condition = self.encode_text(data.embeddings.unsqueeze(1), scene_latent=z)

        # Decode conditioned on cross-attended text
        posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx = self.decode(
            z, data.edge_index, condition
        )

        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        direc = torch.cat((direc_cos, direc_sin), 1)
        direc = F.normalize(direc, dim=-1)

        return pos, size, vel, acttype, direc, laneidx, log_var, mu, pi, pi_logits

    def decoder_only(self, latent, edge_index, text_embeddings=None):
        """
        Decode from a given latent vector, optionally conditioned on text embeddings.
        :param latent: [B, latent_dim] latent vector z
        :param edge_index: graph edge index
        :param text_embeddings: [B, L, embed_size] optional text embeddings
        """
        if text_embeddings is not None:
            condition = self.encode_text(text_embeddings.unsqueeze(1), scene_latent=latent)
        else:
            condition = None

        # Decode conditioned latent
        posx, posy, sizex, sizey, vel, acttype, direc_cos, direc_sin, laneidx = self.decode(
            latent, edge_index, condition
        )

        pos = torch.cat((posx, posy), 1)
        size = torch.cat((sizex, sizey), 1)
        direc = torch.cat((direc_cos, direc_sin), 1)
        direc = F.normalize(direc, dim=-1)

        return pos, size, vel, acttype, direc, laneidx
