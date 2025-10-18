import torch
import math
from einops import einsum, rearrange

def create_matrix_parameter(in_features: int, out_features: int, dtype, device):
    std = math.sqrt(2 / (in_features + out_features))

    return torch.nn.Parameter(
        torch.nn.init.trunc_normal_(
            torch.empty(size=(out_features, in_features), dtype=dtype, device=device),
            mean=0.0,
            std=std,
            a=-3*std,
            b=3*std,
            generator=None))

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = create_matrix_parameter(in_features=in_features, out_features=out_features, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, 'd_out d_in, ... d_in -> ... d_out')
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size=(num_embeddings, embedding_dim), dtype=dtype, device=device),
            mean=0.0,
            std=1,
            a=-3,
            b=3,
            generator=None))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids, :]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps
        self.inv_d_model = 1 / d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sum_squares = einsum(x, x, 'batch seq d_model, batch seq d_model -> batch seq')
        rms = torch.sqrt(self.inv_d_model * sum_squares + self.eps)
        rms = rearrange(rms, 'batch seq -> batch seq ()')
        rms_norm = (x / rms) * self.g

        return rms_norm.to(in_dtype)

class SiLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int|None = None, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            ff_scale = 8 / 3
            ff_multiple_of = 64
            d_ff = ff_multiple_of * round((ff_scale * d_model) / ff_multiple_of)
        self.silu = SiLU()
        self.W1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.W2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_in = self.W1(x)
        gate = self.W3(x)
        gated_silu = self.silu(silu_in) * gate

        return self.W2(gated_silu)

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # Task description indicies start from 1 but tests only pass from 0
        i = torch.arange(0, max_seq_len, device=device)
        k = torch.arange(0, d_k//2, device=device)
        inv_theta_k = 1 / (theta**((2*k)/d_k))
        theta_i_k = einsum(i, inv_theta_k, 'i, k -> i k') # outer product
        cos_theta_i_k = torch.cos(theta_i_k)
        sin_theta_i_k = torch.sin(theta_i_k)
        # Stack block matrices in row-major order and rearrange
        R = rearrange(
            [cos_theta_i_k, -sin_theta_i_k, sin_theta_i_k, cos_theta_i_k],
            '(block_x block_y) i k -> i k block_x block_y',
            block_x=2,
            block_y=2)

        self.register_buffer(
            name="R",
            tensor=R,
            persistent=False
        )
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Split last dim of x into 2d vectors and arrange those along new dimension
        x_split = rearrange(x, '... (k k2) -> ... k k2', k2=2)
        # Retrieve R_i blocks at specified positions
        R_i_blocks = self.R[token_positions,:]
        x_split_rot = einsum(R_i_blocks, x_split, '... bx by, ... by -> ... bx')
        x_rot = rearrange(x_split_rot, '... k k2 -> ... (k k2)', k2=2)

        return x_rot

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_offset = x - x.amax(dim=dim, keepdim=True)
    exp_x = torch.exp(x_offset)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V:torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    QK = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
    QK = QK / math.sqrt(K.shape[-1])
    QK[~mask] = -torch.inf
    QK_softmax = softmax(QK, dim=QK.dim() - 1)

    return einsum(QK_softmax, V, '... queries values, ... values d_v -> ... queries d_v')

class SelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.rope = rope
        self.W_q = create_matrix_parameter(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_k = create_matrix_parameter(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_v = create_matrix_parameter(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_o = create_matrix_parameter(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None, mask: torch.Tensor=None) -> torch.Tensor:
        # TODO: KV-Caching for inference
        # Concat into single matrix
        W_qkv = rearrange([self.W_q, self.W_k, self.W_v], 'concat h_d_k d_in -> (concat h_d_k) d_in')
        QKV_concat = einsum(W_qkv, x, 'concat d_in, ... d_in -> ... concat')
        QKV_concat = rearrange(QKV_concat, '... seq (concat h d_k) -> concat ... h seq d_k', h=self.num_heads, concat=3)
        Q = QKV_concat[0,:]
        K = QKV_concat[1,:]
        V = QKV_concat[2,:]
        seq_len = x.shape[-2]

        if self.rope:
            if token_positions is None:
                raise ValueError("Rope requires token_positions")
            
            K = self.rope(K, token_positions)
            Q = self.rope(Q, token_positions)

        if not mask:
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=bool, device=self.W_k.device))
            rep_dims = x.shape[:-2] # broadcast to batch dimensions
            rep_dims += torch.Size([self.num_heads, 1, 1]) # broadcast to heads
            mask = mask.repeat(rep_dims)
        
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn_concat = rearrange(attn, '... h seq d_v -> ... seq (h d_v)')

        return einsum(self.W_o, attn_concat, 'd_out h_d_v, ... h_d_v -> ... d_out')