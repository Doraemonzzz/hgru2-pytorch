import torch.nn.functional as F
from torch import nn

from .helpers import get_activation_fn, print_params
from .hgru_real_cuda import HgruRealFunction


class Hgru1_real_1d(nn.Module):
    def __init__(
        self,
        embed_dim,
        act_fun="silu",
        expand_ratio=2,
        bias=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.input_proj = nn.Linear(embed_dim, expand_ratio * embed_dim, bias=bias)
        self.forget_gate = nn.Linear(embed_dim, expand_ratio * embed_dim, bias=bias)
        self.output_gate = nn.Linear(embed_dim, expand_ratio * embed_dim, bias=bias)
        self.out_proj = nn.Linear(expand_ratio * embed_dim, embed_dim, bias=bias)
        self.norm = nn.LayerNorm(expand_ratio * embed_dim)
        self.act = get_activation_fn(act_fun)

        self.scan = HgruRealFunction.apply

    def forward(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        input = self.act(self.input_proj(x))
        output_gate = F.sigmoid(self.output_gate(x))
        lambda_ = lower_bound + (1 - lower_bound) * F.sigmoid(self.forget_gate(x))
        input = (1 - lambda_) * input

        output_state = self.scan(input, lambda_)

        output_state = self.norm(output_state * output_gate)

        output = self.out_proj(output_state)

        return output
