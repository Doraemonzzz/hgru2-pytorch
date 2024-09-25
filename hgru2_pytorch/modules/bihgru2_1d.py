import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..gla.inter_chunk_contribution.fn import inter_chunk_onc
from ..gla.intra_chunk_contribution.fn import intra_chunk_onc
from ..gla.recurrent_fuse import fused_recurrent_gla
from ..helpers import get_activation_fn, get_norm_fn, print_module, print_params
from ..hgru_real_cuda import HgruRealFunction


class BiHgru2_1d(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        act_fun="silu",
        uv_act_fun="sigmoid",
        use_norm=True,
        bias=True,
        norm_type="layernorm",
        chunk_size=128,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.expand_ratio = expand_ratio
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        self.out_act = get_activation_fn(uv_act_fun)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(embed_dim)

        self.chunk_size = chunk_size

        if self.expand_ratio < 16:
            self.scan = HgruRealFunction.apply
            self.forward = self.forward_lesshead

    def reverse_scan(self, input, lambda_):
        output_state = self.scan(
            torch.flip(input, dims=[0]),
            torch.flip(lambda_, dims=[0]),
        )

        return torch.flip(output_state, dims=[0])

    def pad(self, x):
        # n, b, h, d
        n, b, h, d = x.shape
        if n % self.chunk_size == 0:
            return x
        else:
            pad = self.chunk_size - n % self.chunk_size
            return F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad)).contiguous()

    def compute(self, Q, K, V, G_K):
        if not self.training:
            dtype = Q.dtype
            V, Q, G_K, K = map(
                lambda x: rearrange(x, "n b h d -> b h n d")
                .to(torch.float32)
                .contiguous(),
                [V, Q, G_K, K],
            )
            o = fused_recurrent_gla(Q, K, V, G_K)
            o = rearrange(o, "b h n d -> n b (h d)").to(dtype)
            return o
        else:
            m = Q.shape[0]
            V, Q, G_K, K = map(lambda x: self.pad(x), [V, Q, G_K, K])
            n, b, h, d = Q.shape
            V, Q, G_K, K = map(
                lambda x: rearrange(
                    x, "(n c) b h d -> b h n c d", c=min(self.chunk_size, n)
                ).contiguous(),
                [V, Q, G_K, K],
            )
            G_V = None
            G_K, G_V, o1 = inter_chunk_onc(Q, K.to(Q.dtype), V, G_K.to(Q.dtype), G_V)
            o2 = intra_chunk_onc(Q, K.to(Q.dtype), V, G_K.to(Q.dtype), G_V)
            o = o1 + o2
            o = rearrange(o, "b h n c d -> (n c) b (h d)")

            return o[:m]

    def forward(self, x, lower_bound=0):
        ## x: n b d
        n, b, d = x.shape

        feature = self.in_proj(x)
        V, Q, F_ = feature.chunk(3, dim=-1)
        V = self.act(V)
        Q = self.out_act(Q)
        F_ = F.sigmoid(F_)
        if type(lower_bound) == int:
            lower_bound = torch.zeros_like(x).to(x)

        # reshape
        # h is num_head, d is head dimension
        V, Q, F_, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [V, Q, F_, lower_bound],
        )

        lambda_ = lower_bound + (1 - lower_bound) * F_

        log_lambda_ = torch.log(lambda_)

        K = 1 - lambda_

        o1 = self.compute(Q, K, V, log_lambda_)
        o2 = torch.flip(
            self.compute(
                torch.flip(Q, dims=[0]),
                torch.flip(K, dims=[0]),
                torch.flip(V, dims=[0]),
                torch.flip(log_lambda_, dims=[0]),
            ),
            dims=[0],
        )
        o = o1 + o2

        if self.use_norm:
            o = self.norm(o)

        # out proj
        output = self.out_proj(o)
        return output

    def forward_lesshead(self, x, lower_bound=0):
        # h = lambda * h + (1 - lambda) * input
        n, b, d = x.shape
        feature = self.in_proj(x)
        input, output_gate, forget_gate = feature.chunk(3, dim=-1)
        input = self.act(input)
        output_gate = self.out_act(output_gate)
        forget_gate = F.sigmoid(forget_gate)
        if type(lower_bound) == int:
            lower_bound = torch.zeros_like(x).to(x)
        # reshape
        input, output_gate, forget_gate, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [input, output_gate, forget_gate, lower_bound],
        )

        # mix
        lower_bound = lower_bound.to(input.dtype)
        lambda_ = lower_bound + (1 - lower_bound) * forget_gate
        input = torch.einsum("... h d, ... h e -> ... h d e", 1 - lambda_, input)
        lambda_ = repeat(lambda_, "... h d -> ... h d e", e=self.expand_ratio)

        # reshape
        input, lambda_ = map(
            lambda x: rearrange(x, "... h d e -> ... (h d e)"), [input, lambda_]
        )

        output_state_forward = self.scan(input, lambda_)
        output_state_reverse = self.reverse_scan(input, lambda_)
        output_state = output_state_forward + output_state_reverse

        # down
        output_state = rearrange(
            output_state,
            "... (h d e) -> ... h d e",
            d=self.expand_ratio,
            e=self.expand_ratio,
        )
        output_state = torch.einsum(
            "... h d e, ... h d -> ... h e", output_state, output_gate
        )
        output_state = rearrange(output_state, "... h e -> ... (h e)")

        # output gate
        if self.use_norm:
            output_state = self.norm(output_state)

        # out proj
        output = self.out_proj(output_state)

        return output

    def extra_repr(self):
        return print_module(self)
