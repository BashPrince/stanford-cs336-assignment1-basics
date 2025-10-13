import torch
import math
from einops import einsum, rearrange

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size=(out_features, in_features), dtype=dtype, device=device),
            mean=0.0,
            std=std,
            a=-3*std,
            b=3*std,
            generator=None))
    
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