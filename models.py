import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import knn_graph
import torch.nn.functional as F

import math
import numpy as np

config = {
    'vocab_size': 513, # tokenizer.vocab_size,
    'n_layers': 1,
    'embed_dim': 512, # 1024
    'max_output_len': 500,
    'n_heads': 16,
    'n_kv_heads': 8,
    'multiple_of': 64,
    'ffn_dim_multiplier': None,
    'norm_eps': 1e-5,
    'max_batch_size': 15,
    'max_seq_len': 2000,
    'device': 'cuda:1',
    'device2': 'cuda:0',
}

class TrgtEncoder(nn.Module):
    def __init__(self,input_dim,embed_dim):
        super(TrgtEncoder,self).__init__()
        self.embed = nn.Embedding(input_dim,embed_dim)
        self.pos_embed = nn.Embedding(60,embed_dim)
    
    def forward(self, x):
        # the_el = (x[:,:,0] == 512.0).nonzero(as_tuple=True)[0]
        # x = x[:,0:the_el[0],:]
        pos = torch.arange(0,60).to(x.get_device())
        emb = self.embed(x) / np.sqrt(513)
        emb += self.pos_embed(pos) / np.sqrt(513)
        emb = torch.sum(emb,dim=-2) / np.sqrt(60)
        return emb

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (m, seq_len, dim) * (m, seq_len, 1) = (m, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # weight is a gain parameter used to re-scale the standardized summed inputs
        # (dim) * (m, seq_len, dim) = (m, seq_Len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        out = self.propagate(edge_index, x=x)
        return out, edge_index

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        tmp = self.mlp(tmp)
        return tmp
    
class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, edges=None,batch=None):
        self.k = x.shape[0]
        if edges is not None:
            edge_index = edges
        else:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)

def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i,2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1,2*i] = np.sin(m_theta)
            R[position, 2*i+1,2*i+1] = np.cos(m_theta)
    return R

def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    
    # theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # (seq_len)
    m = torch.arange(seq_len, device=device)

    # (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # complex numbers in polar, c = R * exp(m * theta), where R = 1:
    # (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex
def apply_rotary_embeddings(x, freqs_complex, device):

    # last dimension pairs of two values represent real and imaginary
    # two consecutive values will become a single complex number

    # (m, seq_len, num_heads, head_dim/2, 2)
    x_in = x.float().reshape(*x.shape[:-1], -1, 2)

    # (m, seq_len, num_heads, head_dim/2)
    x_complex = torch.view_as_complex(x_in)

    # (seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # multiply each complex number
    # (m, seq_len, n_heads, head_dim/2)
    x_rotated = x_complex * freqs_complex

    # convert back to the real number
    # (m, seq_len, n_heads, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (m, seq_len, n_heads, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, device):
        self.cache_k = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim)).to(device)
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim)).to(device)

    def update(self, batch_size, start_pos, xk, xv):
        self.cache_k[:batch_size, start_pos :start_pos + xk.size(1)] = xk # .reshape(xk.shape[0],xk.shape[1],xk.shape[2],xk.shape[3]*xk.shape[4])
        self.cache_v[:batch_size, start_pos :start_pos + xv.size(1)] = xv

    def get(self, batch_size, start_pos, seq_len):
        keys = self.cache_k[:batch_size,  :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        return keys, values
    
def repeat_kv(x, n_rep):
    
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (m, seq_len, n_kv_heads, 1, head_dim)
        # --> (m, seq_len, n_kv_heads, n_rep, head_dim)
        # --> (m, seq_len, n_kv_heads * n_rep, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config['n_heads']
        self.n_kv_heads = config['n_kv_heads']
        self.dim = config['embed_dim']
        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        self.n_heads_q = self.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.cache = KVCache(
            max_batch_size=config['max_batch_size'],
            max_seq_len=config['max_seq_len'],
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=config['device']
        )

    def forward(self, x, start_pos, freqs_complex):

        # seq_len is always 1 during inference
        batch_size, seq_len, _ = x.shape

        # (m, seq_len, dim)
        xq = self.wq(x)

        # (m, seq_len, h_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (m, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (m, seq_len, h_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (m, seq_len, num_head, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)

        # (m, seq_len, h_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # replace the entry in the cache
        self.cache.update(batch_size, start_pos, xk, xv)

        # (m, seq_len, h_kv, head_dim)
        keys, values = self.cache.get(batch_size, start_pos, seq_len)

        # (m, seq_len, h_kv, head_dim) --> (m, seq_len, n_heads, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (m, n_heads, seq_len, head_dim)
        # seq_len is 1 for xq during inference
        xq = xq.transpose(1, 2)

        # (m, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (m, n_heads, seq_len_q, head_dim) @ (m, n_heads, head_dim, seq_len) -> (m, n_heads, seq_len_q, seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # (m, n_heads, seq_len_q, seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (m, n_heads, seq_len_q, seq_len) @ (m, n_heads, seq_len, head_dim) -> (m, n_heads, seq_len_q, head_dim)
        output = torch.matmul(scores, values)

        # ((m, n_heads, seq_len_q, head_dim) -> (m, seq_len_q, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        # (m, seq_len_q, dim)
        return self.wo(output)
    

def sigmoid(x, beta=1):
    return 1 / (1 + torch.exp(-x * beta))

def swiglu(x, beta=1):
    return x * sigmoid(x, beta)

class FeedForward(nn.Module):
    def __init__(self, config):

        super().__init__()

        hidden_dim = 4 * config['embed_dim']
        hidden_dim = int(2 * hidden_dim / 3)

        if config['ffn_dim_multiplier'] is not None:
            hidden_dim = int(config['ffn_dim_multiplier'] * hidden_dim)

        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = config['multiple_of'] * ((hidden_dim + config['multiple_of'] - 1) // config['multiple_of'])

        self.w1 = nn.Linear(config['embed_dim'], hidden_dim, bias=False)
        self.w2 = nn.Linear(config['embed_dim'], hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config['embed_dim'], bias=False)

    def forward(self, x: torch.Tensor):
        # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
        swish = swiglu(self.w1(x))
        # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
        x_V = self.w2(x)

        # (m, seq_len, hidden_dim)
        x = swish * x_V

        # (m, seq_len, hidden_dim) --> (m, seq_len, dim)
        return self.w3(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config['n_heads']
        self.dim = config['embed_dim']
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)

        # rms before attention block
        self.attention_norm = RMSNorm(self.dim, eps=config['norm_eps'])

        # rms before  feed forward block
        self.ffn_norm = RMSNorm(self.dim, eps=config['norm_eps'])

    def forward(self, x, start_pos, freqs_complex):

        # (m, seq_len, dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex)
        # (m, seq_len, dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config['vocab_size']
        self.n_layers = config['n_layers']
        self.tok_embeddings = nn.Embedding(self.vocab_size, config['embed_dim'])
        self.head_dim = config['embed_dim'] // config['n_heads']
        self.n_layers = 2
        self.chunk = int(config['max_seq_len'] / config['max_output_len'])
        
        self.gru = nn.GRU(config['embed_dim'],config['embed_dim'],self.n_layers, batch_first=True,dropout=0.2)
        self.reduce_fc = nn.Linear(config['embed_dim']*self.chunk,config['embed_dim'])
        self.leakyrelu = nn.LeakyReLU()

        self.layers = nn.ModuleList()
        for layer_id in range(config['n_layers']):
            self.layers.append(DecoderBlock(config))

        self.norm = RMSNorm(config['embed_dim'], eps=config['norm_eps'])
        self.output = nn.Linear(config['embed_dim'], self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.head_dim, config['max_seq_len'] * 2, device=(config['device']))
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, config['embed_dim']).zero_() #.to(self.get_device())
        return hidden

    def forward(self, tokens, start_pos):
        # (m, seq_len)
        batch_size, seq_len, wrd_len = tokens.shape

        # (m, seq_len) -> (m, seq_len, embed_dim)
        h = self.tok_embeddings(tokens.long())
        h = torch.sum(h,dim=-2)

        # (seq_len, (embed_dim/n_heads)/2]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers
        # (m, seq_len, dim)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        
        h_cell = self.init_hidden(15)
        
        out2 = None
        for x_chunk in h.chunk(config['max_output_len'],dim=1):
            out, h = self.gru(x_chunk,h_cell)
            out = self.leakyrelu(self.reduce_fc(out.reshape(out.shape[0],1,out.shape[1]*out.shape[2])))
            out = self.norm(out)
            if out2 == None:
                out2 = out
            else:
                out2 = torch.concat([out2,out],dim=1)
        # (m, seq_len, vocab_size)
        output = self.output(out2).float()
        del h_cell
        return output

# model = Transformer(config).to(config['device'])
# res = model.forward(test_set['input_ids'].to(config['device']), 0)
# print(res.size())