# coding=utf-8
""" Hgrn configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class Hgrn2Config(PretrainedConfig):
    model_type = "hgrn2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        vocab_size=50272,
        use_cache=True,
        init_std=0.02,
        # model config
        decoder_embed_dim=1024,
        expand_ratio=2,
        decoder_layers=24,
        add_bos_token=False,
        in_act="none",
        out_act="silu",
        causal=True,
        glu_act="none",
        glu_dim=2816,
        bias=False,
        norm_type="simplermsnorm",
        no_scale_embedding=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # hf origin
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.init_std = init_std
        # add
        self.decoder_embed_dim = decoder_embed_dim
        self.expand_ratio = expand_ratio
        self.decoder_layers = decoder_layers
        self.add_bos_token = add_bos_token
        self.in_act = in_act
        self.out_act = out_act
        self.causal = causal
        self.glu_act = glu_act
        self.glu_dim = glu_dim
        self.bias = bias
        self.norm_type = norm_type
        self.no_scale_embedding = no_scale_embedding
