# coding=utf-8
""" PyTorch Hgrn2 model."""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from ..gla.inter_chunk_contribution.fn import inter_chunk_onc
from ..gla.intra_chunk_contribution.fn import intra_chunk_onc
from ..gla.recurrent_fuse import fused_recurrent_gla
from ..helpers import get_activation_fn, get_norm_fn, print_module, print_params
from .configuration_hgrn2 import Hgrn2Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Hgrn2Config"


class Hgru2(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        in_act="none",
        out_act="silu",
        use_norm=True,
        bias=False,
        norm_type="simplermsnorm",
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.expand_ratio = expand_ratio
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_act = get_activation_fn(in_act)
        self.out_act = get_activation_fn(out_act)
        self.use_norm = use_norm
        self.norm = get_norm_fn(norm_type)(embed_dim)

        self.chunk_size = 128

    def forward(self, x, lower_bound=0):
        ## x: n b d
        n, b, d = x.shape
        feature = self.in_proj(x)
        V, Q, F_ = feature.chunk(3, dim=-1)
        V = self.in_act(V)
        Q = self.out_act(Q)
        F_ = F.sigmoid(F_)

        # reshape
        # h is num_head, d is head dimension
        V, Q, F_, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [V, Q, F_, lower_bound],
        )

        lambda_ = lower_bound + (1 - lower_bound) * F_

        log_lambda_ = torch.log(lambda_)

        K = 1 - lambda_

        if self.training:
            V, Q, G_K, K = map(
                lambda x: rearrange(
                    self.pad(x), "(n c) b h d -> b h n c d", c=self.chunk_size
                ).contiguous(),
                [V, Q, log_lambda_, K],
            )
            G_V = None
            G_K, G_V, o1 = inter_chunk_onc(Q, K, V, G_K, G_V)
            o2 = intra_chunk_onc(Q, K, V, G_K, G_V)
            o = o1 + o2
            o = rearrange(o, "b h n c d -> (n c) b (h d)")
        else:
            V, Q, G_K, K = map(
                lambda x: rearrange(x, "n b h d -> b h n d")
                .to(torch.float32)
                .contiguous(),
                [V, Q, log_lambda_, K],
            )
            o = fused_recurrent_gla(Q, K, V, G_K)
            o = rearrange(o, "b h n d -> n b (h d)").to(x.dtype)

        if self.use_norm:
            o = self.norm(o)

        # out proj
        output = self.out_proj(o[:n])

        return output

    def pad(self, x):
        # n, b, h, d
        n, b, h, d = x.shape
        if n % self.chunk_size == 0:
            return x
        else:
            pad = self.chunk_size - n % self.chunk_size
            return F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad)).contiguous()

    def extra_repr(self):
        return print_module(self)


class GLU(nn.Module):
    def __init__(self, d1, d2, act_fun, bias=False):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = get_activation_fn(act_fun)

    def forward(self, x):
        o1 = self.act_fun(self.l1(x))
        o2 = self.l2(x)
        output = o1 * o2
        output = self.l3(output)

        return output


class Hgrn2DecoderLayer(nn.Module):
    def __init__(self, config: Hgrn2Config):
        super().__init__()
        self.embed_dim = config.decoder_embed_dim
        ##### token mixer
        self.token_mixer = Hgru2(
            embed_dim=self.embed_dim,
            expand_ratio=config.expand_ratio,
            in_act=config.in_act,
            out_act=config.out_act,
            bias=config.bias,
            norm_type=config.norm_type,
        )
        self.token_norm = get_norm_fn(config.norm_type)(self.embed_dim)

        ##### channel mixer
        self.glu_act = config.glu_act
        self.glu_dim = config.glu_dim
        self.channel_mixer = GLU(
            self.embed_dim, self.glu_dim, self.glu_act, bias=config.bias
        )
        self.channel_norm = get_norm_fn(config.norm_type)(self.embed_dim)

    def forward(
        self,
        x,
        padding_mask: Optional[torch.Tensor] = None,
        lower_bound: Optional[torch.Tensor] = None,
    ):
        # current does not support padding_mask!
        x = self.token_mixer(self.token_norm(x), lower_bound) + x
        x = self.channel_mixer(self.channel_norm(x)) + x

        outputs = x

        return outputs, None


HGRN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Hgrn2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    HGRN2_START_DOCSTRING,
)
class Hgrn2PreTrainedModel(PreTrainedModel):
    config_class = Hgrn2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Hgrn2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Hgrn2Model):
            module.gradient_checkpointing = value


@dataclass
class Hgrn2ModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    cache_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


HGRN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    HGRN2_START_DOCSTRING,
)
class Hgrn2Model(Hgrn2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Hgrn2DecoderLayer`]

    Args:
        config: Hgrn2Config
    """

    def __init__(self, config: Hgrn2Config):
        super().__init__(config)
        # hf origin
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        # params
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.decoder_embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Hgrn2DecoderLayer(config) for i in range(config.decoder_layers)]
        )
        self.final_norm = get_norm_fn(config.norm_type)(config.decoder_embed_dim)
        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = (
            1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)
        )
        self.num_layers = config.decoder_layers
        self.lower_bounds = nn.Parameter(
            torch.ones(self.num_layers, self.embed_dim), requires_grad=True
        )

        # v1
        # slope = torch.log(self._build_slope_tensor(self.embed_dim))
        # self.register_buffer("lower_bound", slope)

        # Initialize weights and apply final processing
        self.post_init()

    def extra_repr(self):
        return print_module(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(HGRN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        padding_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if (
            not self.training
            and padding_mask != None
            and padding_mask.eq(self.padding_idx)
        ):
            raise ValueError(
                "During the inference stage, attn_padding_mask should be either None or should not include the pad token."
            )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            # !!! use embed_scale
            inputs_embeds = self.embed_scale * self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        cache_values = ()

        # lower bound
        lower_bounds = self.lower_bounds
        lower_bounds = F.softmax(lower_bounds, dim=0)
        lower_bounds = torch.cumsum(lower_bounds, dim=0)
        lower_bounds -= lower_bounds[0, ...].clone()

        # b, n, d -> n, b, d
        hidden_states = hidden_states.transpose(1, 0)

        for idx, layer in enumerate(self.layers):
            # v1
            # lower_bound = self.lower_bound / (idx + 1)
            lower_bound = lower_bounds[idx]

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    padding_mask,
                    lower_bound,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    padding_mask,
                    lower_bound,
                )

            hidden_states = layer_outputs[0]

            # tbd
            cache_values += (layer_outputs[1],)

        hidden_states = self.final_norm(hidden_states)

        # n, b, d -> b, n, d
        hidden_states = hidden_states.transpose(1, 0)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_values] if v is not None)
        return Hgrn2ModelOutputWithPast(
            last_hidden_state=hidden_states, cache_values=cache_values
        )

    @staticmethod
    def _build_slope_tensor(d: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        # h, 1, 1
        slopes = torch.tensor(get_slopes(d))

        return slopes


class Hgrn2ForCausalLM(Hgrn2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Hgrn2Model(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(
            config.decoder_embed_dim, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(HGRN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Hgrn2ForCausalLM

        >>> model = Hgrn2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            padding_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.cache_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attn_padding_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({})
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`Hgrn2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    HGRN2_START_DOCSTRING,
)
class Hgrn2ForSequenceClassification(Hgrn2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Hgrn2Model(config)
        self.score = nn.Linear(config.decoder_embed_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(HGRN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            padding_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            hidden_states=outputs.hidden_states,
        )
