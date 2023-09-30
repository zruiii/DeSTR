from typing import Optional, Tuple, Union, List

import pdb
import torch
from torch import nn, Tensor
from torch.nn import LayerNorm
from torch_geometric.typing import OptTensor
from tsl.nn.blocks.encoders import MLP
from tsl.nn.functional import sparse_softmax
from tsl.nn.layers import PositionalEncoding

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, OptPairTensor
from torch_scatter import scatter
from torch_scatter.utils import broadcast
from torch.nn import LayerNorm, functional as F
from torch_geometric.nn import inits

import math
class StaticGraphEmbedding(nn.Module):
    r"""Creates a table of embeddings with the specified size.

    Args:
        n_tokens (int): Number of elements for which to store an embedding.
        emb_size (int): Size of the embedding.
        initializer (str or Tensor): Initialization methods.
            (default :obj:`'uniform'`)
        requires_grad (bool): Whether to compute gradients for the embeddings.
            (default :obj:`True`)
        bind_to (nn.Module, optional): Bind the embedding to a nn.Module for
            lazy init. (default :obj:`None`)
        infer_tokens_from_pos (int): Index of the element of input data from
            which to infer the number of embeddings for lazy init.
            (default :obj:`0`)
        dim (int): Token dimension. (default :obj:`-2`)
    """

    def __init__(self, n_tokens: int, emb_size: int,
                 initializer: Union[str, Tensor] = 'uniform',
                 requires_grad: bool = True,
                 bind_to: Optional[nn.Module] = None,
                 infer_tokens_from_pos: int = 0,
                 dim: int = -2):
        super(StaticGraphEmbedding, self).__init__()
        assert emb_size > 0
        self.n_tokens = int(n_tokens)
        self.emb_size = int(emb_size)
        self.dim = int(dim)
        self.infer_tokens_from_pos = infer_tokens_from_pos

        if isinstance(initializer, Tensor):
            self.initializer = "from_values"
            self.register_buffer('_default_values', initializer.float())
        else:
            self.initializer = initializer
            self.register_buffer('_default_values', None)

        if self.n_tokens > 0:
            self.emb = nn.Parameter(Tensor(self.n_tokens, self.emb_size),
                                    requires_grad=requires_grad)
        else:
            assert isinstance(bind_to, nn.Module)
            self.emb = nn.parameter.UninitializedParameter(
                requires_grad=requires_grad)
            bind_to._hook = bind_to.register_forward_pre_hook(
                self.initialize_parameters)

        self.reset_parameters()

    def reset_parameters(self):
        if self.n_tokens > 0:
            if self.initializer == 'from_values':
                self.emb.data = self._default_values.data
            if self.initializer == 'glorot':
                inits.glorot(self.emb)
            elif self.initializer == 'uniform' or self.initializer is None:
                inits.uniform(self.emb_size, self.emb)
            elif self.initializer == 'kaiming_normal':
                nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
            elif self.initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.emb, fan=self.emb_size,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(f"Embedding initializer '{self.initializer}'"
                                   " is not supported")

    def extra_repr(self) -> str:
        return f"n_tokens={self.n_tokens}, embedding_size={self.emb_size}"

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.emb, torch.nn.parameter.UninitializedParameter):
            self.n_tokens = input[self.infer_tokens_from_pos].size(self.dim)
            self.emb.materialize((self.n_tokens, self.emb_size))
            self.reset_parameters()
        module._hook.remove()
        delattr(module, '_hook')

    def forward(self, expand: Optional[List] = None,
                token_index: OptTensor = None,
                tokens_first: bool = True):
        """"""
        emb = self.emb if token_index is None else self.emb[token_index]
        if not tokens_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [1 if d > 0 else shape.pop(0 if tokens_first else -1)
                for d in expand]
        return emb.view(*view).expand(*expand)


class PositionalEncoder(nn.Module):

    def __init__(self, in_channels, out_channels,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None):
        super(PositionalEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels, out_channels, out_channels,
                       n_layers=n_layers, activation='relu')
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = nn.Embedding(n_nodes, out_channels)
        else:
            self.register_parameter('node_emb', None)
        
        self.n_nodes = n_nodes

    def forward(self, x, node_emb=None):
        """
        x: [B, L, C]
        out: [B, L, N, C]
        """
        B, L, C = x.shape
        node_index = torch.arange(self.n_nodes, device=x.device)
        node_emb = self.node_emb(node_index)[None, None, :, :]

        x = self.lin(x)
        x = self.activation(x.unsqueeze(-2) + node_emb)
        out = self.mlp(x)
        out = self.positional(out)
        return out

class AdditiveAttention(MessagePassing):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 dim: int = -2,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=dim, **kwargs)

        self.output_size = output_size
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size

        self.msg_size = msg_size or self.output_size
        self.msg_layers = msg_layers

        assert reweight in ['softmax', 'l1', None]
        self.reweight = reweight

        self.root_weight = root_weight
        self.dropout = dropout

        # key bias is discarded in softmax
        self.lin_src = Linear(self.src_size, self.output_size,
                              weight_initializer='glorot',
                              bias_initializer='zeros')
        self.lin_tgt = Linear(self.tgt_size, self.output_size,
                              weight_initializer='glorot', bias=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size,
                                   bias=False)
        else:
            self.register_parameter('lin_skip', None)

        self.msg_nn = nn.Sequential(
            nn.PReLU(init=0.2),
            MLP(self.output_size, self.msg_size, self.output_size,
                n_layers=self.msg_layers, dropout=self.dropout,
                activation='prelu')
        )

        if self.reweight == 'softmax':
            self.msg_gate = nn.Linear(self.output_size, 1, bias=False)
        else:
            self.msg_gate = nn.Sequential(nn.Linear(self.output_size, 1),
                                          nn.Sigmoid())

        if norm:
            self.norm = LayerNorm(self.output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_tgt.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()

    def forward(self, x: PairTensor, edge_index: Adj, mask: OptTensor = None):
        # if query/key not provided, defaults to x (e.g., for self-attention)
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        N_src, N_tgt = x_src.size(self.node_dim), x_tgt.size(self.node_dim)

        msg_src = self.lin_src(x_src)
        msg_tgt = self.lin_tgt(x_tgt)

        msg = (msg_src, msg_tgt)

        # propagate_type: (msg: PairTensor, mask: OptTensor)
        out = self.propagate(edge_index, msg=msg, mask=mask,
                             size=(N_src, N_tgt))

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def normalize_weights(self, weights, index, num_nodes, mask=None):
        # mask weights
        if mask is not None:
            fill_value = float("-inf") if self.reweight == 'softmax' else 0.
            weights = weights.masked_fill(torch.logical_not(mask), fill_value)
        # eventually reweight
        if self.reweight == 'l1':
            expanded_index = broadcast(index, weights, self.node_dim)
            weights_sum = scatter(weights, expanded_index, self.node_dim,
                                  dim_size=num_nodes, reduce='sum')
            weights_sum = weights_sum.index_select(self.node_dim, index)
            weights = weights / (weights_sum + 1e-5)
        elif self.reweight == 'softmax':
            weights = sparse_softmax(weights, index, num_nodes=num_nodes,
                                     dim=self.node_dim)
        return weights

    def message(self, msg_j: Tensor, msg_i: Tensor, index, size_i,
                mask_j: OptTensor = None) -> Tensor:
        msg = self.msg_nn(msg_j + msg_i)
        gate = self.msg_gate(msg)
        alpha = self.normalize_weights(gate, index, size_i, mask_j)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = alpha * msg
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.output_size}, '
                f'dim={self.node_dim}, '
                f'root_weight={self.root_weight})')


class TemporalAdditiveAttention(AdditiveAttention):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('dim', 1)
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         msg_size=msg_size,
                         msg_layers=msg_layers,
                         root_weight=root_weight,
                         reweight=reweight,
                         dropout=dropout,
                         norm=norm,
                         **kwargs)

    def forward(self, x: PairTensor, mask: OptTensor = None,
                temporal_mask: OptTensor = None,
                causal_lag: Optional[int] = None):
        # x: [b s * c]    query: [b l * c]    key: [b s * c]
        # mask: [b s * c]    temporal_mask: [l s]
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        l, s = x_tgt.size(self.node_dim), x_src.size(self.node_dim)
        i = torch.arange(l, dtype=torch.long, device=x_src.device)
        j = torch.arange(s, dtype=torch.long, device=x_src.device)

        # compute temporal index, from j to i
        if temporal_mask is None and isinstance(causal_lag, int):
            temporal_mask = tuple(torch.tril_indices(l, l, offset=-causal_lag,
                                                     device=x_src.device))
        if temporal_mask is not None:
            assert temporal_mask.size() == (l, s)
            i, j = torch.meshgrid(i, j)
            edge_index = torch.stack((j[temporal_mask], i[temporal_mask]))
        else:
            edge_index = torch.cartesian_prod(j, i).T

        return super(TemporalAdditiveAttention, self).forward(x, edge_index,
                                                              mask=mask)


class TemporalGraphAdditiveAttention(MessagePassing):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 temporal_self_attention: bool = True,
                 mask_temporal: bool = True,
                 mask_spatial: bool = True,
                 norm: bool = True,
                 dropout: float = 0.,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TemporalGraphAdditiveAttention, self).__init__(node_dim=-2,
                                                             **kwargs)

        # store dimensions
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size
        self.output_size = output_size
        self.msg_size = msg_size or self.output_size

        self.mask_temporal = mask_temporal
        self.mask_spatial = mask_spatial

        self.root_weight = root_weight
        self.dropout = dropout

        if temporal_self_attention:
            self.self_attention = TemporalAdditiveAttention(
                input_size=input_size,
                output_size=output_size,
                msg_size=msg_size,
                msg_layers=msg_layers,
                reweight=reweight,
                dropout=dropout,
                root_weight=False,
                norm=False
            )
        else:
            self.register_parameter('self_attention', None)

        self.cross_attention = TemporalAdditiveAttention(input_size=input_size,
                                                         output_size=output_size,
                                                         msg_size=msg_size,
                                                         msg_layers=msg_layers,
                                                         reweight=reweight,
                                                         dropout=dropout,
                                                         root_weight=False,
                                                         norm=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size,
                                   bias_initializer='zeros')
        else:
            self.register_parameter('lin_skip', None)

        if norm:
            self.norm = LayerNorm(output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        if self.self_attention is not None:
            self.self_attention.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(self, x: OptPairTensor,
                edge_index: Adj, edge_weight: OptTensor = None,
                mask: OptTensor = None):
        """
        x: [B, L, N, H]
        edge_index: [2, E]
        """
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        n_src, n_tgt = x_src.size(-2), x_tgt.size(-2)

        # propagate query, key and value
        out = self.propagate(x=(x_src, x_tgt),
                             edge_index=edge_index, edge_weight=edge_weight,
                             mask=mask if self.mask_spatial else None,
                             size=(n_src, n_tgt))

        if self.self_attention is not None:
            s, l = x_src.size(1), x_tgt.size(1)
            if s == l:
                attn_mask = ~torch.eye(l, l, dtype=torch.bool,
                                       device=x_tgt.device)
            else:
                attn_mask = None
            temp = self.self_attention(x=(x_src, x_tgt),
                                       mask=mask if self.mask_temporal else None,
                                       temporal_mask=attn_mask)
            out = out + temp

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_weight: OptTensor, mask_j: OptTensor) -> Tensor:
        # [batch, steps, edges, channels]

        out = self.cross_attention((x_j, x_i), mask=mask_j)

        if edge_weight is not None:
            out = out * edge_weight.view(-1, 1)
        return out


class SPIN(nn.Module):
    def __init__(self, input_size: int,                         # 输入信号维度
                 hidden_size: int,                              # 隐层维度
                 n_nodes: int,                                  # 节点数目
                 u_size: Optional[int] = None,                  # 时间特征维度
                 output_size: Optional[int] = None,           
                 temporal_self_attention: bool = True,
                 reweight: Optional[str] = 'softmax',
                 n_layers: int = 2,
                 eta: int = 3,
                 message_layers: int = 1):
        super(SPIN, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        self.u_enc = PositionalEncoder(in_channels=u_size,
                                       out_channels=hidden_size,
                                       n_layers=2,
                                       n_nodes=n_nodes)

        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)

        self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            encoder = TemporalGraphAdditiveAttention(
                input_size=hidden_size,
                output_size=hidden_size,
                msg_size=hidden_size,
                msg_layers=message_layers,
                temporal_self_attention=temporal_self_attention,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=l < eta,
                norm=True,
                root_weight=True,
                dropout=0.0
            )
            readout = MLP(hidden_size, hidden_size, output_size,
                          n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def forward(self, xt, xg=None, t=None, nids=None, g=None):
    # def forward(self, x: Tensor, u: Tensor, edge_index: Tensor):
        """
        x: [B, N, L]
        u: [B, L, C]  时间表征
        """
        target_nodes = slice(None)

        # Whiten missing values
        x = xt.transpose(2, 1).unsqueeze(-1)               # [B, N, L, 1]
        mask = ~torch.isnan(x)
        x = x * mask
        x[torch.isnan(x)] = 0

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        # q = self.u_enc(u, node_index=node_index)
        q = self.u_enc(t)                       # [B, L, N, C]

        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        # h = torch.where(mask.bool(), h, q)
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        node_index = None
        src, dst = g.edges()
        edge_index = torch.concat((src[None, :], dst[None, :]))
        imputations = []
        for l in range(self.n_layers):
            if l == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(token_index=node_index)
                masked = self.mask_emb(token_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)

            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[l](x) * mask  # skip connection for valid x
            h = self.encoder[l](h, edge_index, mask=mask)
            # Read from H to get imputations
            target_readout = self.readout[l](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)
        x_hat = x_hat.squeeze().transpose(2, 1)         

        return x_hat
    
    
    def get_loss(self, pred, org, mask, mask_imp_weight=0.5, obs_rec_weight=0.5):
        """
        pred: [B, N, L]     重构序列
        org:  [B, N, L]     完整输入
        mask: [B, N, L]     imputation mask
        """
        org_mask = ~torch.isnan(org)
        rec_mask = org_mask & ~mask
        obs_rec_loss = torch.abs(pred[rec_mask] - org[rec_mask]).mean()
        mask_imp_loss = torch.abs(pred[mask] - org[mask]).mean()
        if mask_imp_weight > 0 and obs_rec_weight > 0:      # Hybrid Loss
            loss = mask_imp_loss * mask_imp_weight + obs_rec_loss * obs_rec_weight
        elif mask_imp_weight == 0:                  # Only Masked Imputation
            loss = obs_rec_loss     
        elif obs_rec_weight == 0:                   # Only Observation Reconstruct
            loss = mask_imp_loss
        else:
            raise NotImplementedError
        
        return loss
    

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
                        options=[32, 64, 128, 256])
        parser.add_argument('--u-size', type=int, default=None)
        parser.add_argument('--output-size', type=int, default=None)
        parser.add_argument('--temporal-self-attention', type=bool,
                            default=True)
        parser.add_argument('--reweight', type=str, default='softmax')
        parser.add_argument('--n-layers', type=int, default=4)
        parser.add_argument('--eta', type=int, default=3)
        parser.add_argument('--message-layers', type=int, default=1)
        return parser



if __name__ == "__main__":
    xt = torch.randn(32, 10, 24)            # B, N, L
    idx = torch.randn(32, 10, 24)
    xt[idx < 0] = torch.nan

    u = torch.randn(32, 24, 5)
    
    edge_index = torch.tensor([[1,2], [3,4], [5,6]]).T
    model = SPIN(input_size=1, hidden_size=64, n_nodes=10, u_size=5)
    imputations = model(xt, u, edge_index)
    print(imputations.shape)
    # loss = model.loss
    # print(imputations.shape, loss.item())

