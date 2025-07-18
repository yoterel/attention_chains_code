import torch
import numpy as np
from torch import nn
from typing import Optional, Union, Tuple
from transformers.models.vit.modeling_vit import ViTSdpaSelfAttention
import math

# Define a recursive monkey patching function
def replace_attn(module, config):
    module_output = module
    # ViTSdpaSelfAttention doesnt exist in transformers 4.51.0 (4.49.0 works)
    # perhaps just include it directly...
    if isinstance(module, ViTSdpaSelfAttention):
        module_output = CustomAttention(config,
                                        module.query, 
                                        module.key, 
                                        module.value, 
                                        module.dropout)
    for name, child in module.named_children():
        module_output.add_module(name, replace_attn(child, config))
    del module

    return module_output

class DummyLayer(nn.Module):
    def forward(self, x):
        return x

class CustomAttentionBase(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class CustomAttention(CustomAttentionBase):
    def __init__(self, config, query, key, value, dropout) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.query.load_state_dict(query.state_dict())
        self.key.load_state_dict(key.state_dict())
        self.value.load_state_dict(value.state_dict())
        self.dropout.load_state_dict(dropout.state_dict())
        # Dummy layers for hooking
        self.dummy_before_softmax = DummyLayer()
        self.dummy_after_softmax = DummyLayer()


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`ViTSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # context_layer = torch.nn.functional.scaled_dot_product_attention(
        context_layer = self.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None

    # Efficient implementation equivalent to the following:
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias

        attn_weight = self.dummy_before_softmax(attn_weight)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dummy_after_softmax(attn_weight)

        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

def collect_attentions(attentions):
    all_attentions = []
    with torch.no_grad():
        for attention in attentions:
            all_attentions.append(attention.cpu())
    return all_attentions


class ManipulateAttention:
    def __init__(self, 
                model, 
                drop_mode_='stst_col_random',
                q_=0.05, 
                just_save=False,
                layers_to_use=None,
                weights=None):
        self.q = q_
        self.drop_mode = drop_mode_
        self.layers_to_use = layers_to_use
        self.model = model
        self.n_layers = len(self.model.vit.encoder.layer)
        self.hooks = []
        self.c = 0
        for name, module in self.model.named_modules():
            if just_save:
                if "dummy_before_softmax" in name:
                    self.c += 1
                    h1 = module.register_forward_hook(self.save_attention)
                    self.hooks.append(h1)
        self.attentions = []
        self.vss = []
        self.lambdas = []
        self.cur_layer = 0
        self.per_layer_weights = weights  # (n_layers_to_use, n_heads)

    def save_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
    
    def collect_attentions(self, input_tensor, return_output=False):
        with torch.no_grad():
            output = self.model(input_tensor)
        all_attentions = collect_attentions(self.attentions)
        if return_output:
            return None, output
        else:
            return all_attentions
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()