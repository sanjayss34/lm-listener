from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

from einops.layers.torch import Rearrange
import numpy as np

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoModelForCausalLM, AutoConfig

import models.pos_encoding as pos_encoding

class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                extra_dim=[],
                top_p=None,
                gpt2_config=None,
                **kwargs):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, extra_dim=extra_dim, gpt2_config=None)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        self.top_p = top_p

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature, extra_inputs=None):
        feat = self.trans_base(idxs, clip_feature, extra_inputs=extra_inputs)
        logits = self.trans_head(feat)
        return logits

    def sample(self, clip_feature, if_categorial=False, extra_inputs=None, extra_inputs_mask=None):
        xs = torch.zeros((clip_feature.shape[0], 0)).to(clip_feature.device).long()
        length = self.block_size
        if extra_inputs is not None:
            length = min([length]+[inp.shape[1] for inp in extra_inputs])
        for k in range(length):
            if k == 0:
                x = []
            else:
                x = xs
            if extra_inputs is not None:
                logits = self.forward(x, clip_feature, extra_inputs=[inp[:,:k+1,:] for inp in extra_inputs])
            else:
                logits = self.forward(x, clip_feature)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx[0] == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            elif self.top_p is not None:
                sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumulative_probs < self.top_p
                nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1]+(1,)), nucleus[..., :-1]], dim=-1)
                sorted_probs[~nucleus] = 0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                dist = Categorical(sorted_probs)
                idx = dist.sample()
                idx = indices.gather(-1, idx.unsqueeze(-1))
                if idx[0] == self.num_vq:
                    break
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == self.block_size - 1:
                return xs[:, :-1]
        return xs

class GPT2MotionTransformer(nn.Module):
    def __init__(self,
        num_vq=1024,
        num_input_vq=0,
        model_name="gpt2-large",
        top_p=None,
        extra_input_dim={},
        output_layers=0,
        freeze_lm=False,
        not_pretrained=False,
        gradient_checkpointing=False,
        predict_input_vq=False,
        use_lora=False,
        **kwargs
    ):
        super().__init__()
        if not_pretrained:
            config = AutoConfig.from_pretrained(model_name)
            self.gpt = AutoModelForCausalLM.from_config(config)
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.text_vocab_size = self.gpt.config.vocab_size
        self.gpt.resize_token_embeddings(self.text_vocab_size+num_vq+num_input_vq+1)
        if not hasattr(self.gpt, "transformer"):
            self.gpt.transformer = self.gpt.model
            self.gpt.transformer.wte = self.gpt.transformer.decoder.embed_tokens
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.gpt.gradient_checkpointing_enable()
        if hasattr(self.gpt.config, "n_embd"):
            self.hidden_size = self.gpt.config.n_embd
        else:
            self.hidden_size = self.gpt.config.hidden_size
        if freeze_lm:
            for name, param in self.gpt.named_parameters():
                param.requires_grad = False
        if freeze_lm or use_lora:
            self.vq_embedding = nn.Embedding(num_vq+num_input_vq+1, self.hidden_size)
        output_size = num_vq
        if predict_input_vq:
            output_size += num_input_vq
        if output_layers == 0:
            self.output_layer = nn.Linear(self.gpt.config.n_embd, output_size)
        else:
            self.output_layer = nn.Sequential(
                *([
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.ReLU()
                ]*(output_layers)),
                nn.Linear(self.hidden_size // 2, output_size)
            )
        if len(extra_input_dim) > 0:
            self.extra_input_layers = nn.ModuleDict()
            for key, value in extra_input_dim.items():
                self.extra_input_layers.update({key: nn.Linear(value, self.hidden_size)})
            # self.extra_input_layers = nn.ModuleList([nn.Linear(dim, self.hidden_size) for dim in extra_input_dim])
            # self.extra_input_layer = nn.Linear(extra_input_dim[0], self.hidden_size)
        if use_lora:
            model_names = sorted(list(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.keys()), key=lambda x: -len(x))
            modules = None
            for name in model_names:
                if name in model_name:
                    modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[name]
            assert modules is not None
            self.lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            self.gpt = get_peft_model(self.gpt, self.lora_config)
        self.num_vq = num_vq
        self.num_input_vq = num_vq
        self.top_p = top_p
    def forward(self, input_ids, attention_mask, input_embeds=None, predict_input_vq=False):
        if input_embeds is not None:
            bs = input_embeds.shape[0]
            t = input_embeds.shape[1]
            if hasattr(self, "vq_embedding"):
                input_embeds = torch.where(
                    (input_ids >= self.text_vocab_size).unsqueeze(-1).repeat(1, 1, input_embeds.shape[-1]),
                    input_embeds+self.vq_embedding((input_ids-self.text_vocab_size).clamp(min=0)),
                    input_embeds
                )
            outputs = self.gpt(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True, use_cache=False if self.gradient_checkpointing else None)
        else:
            bs = input_ids.shape[0]
            t = input_ids.shape[1]
            outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False if self.gradient_checkpointing else None)
        logits = self.output_layer(outputs.hidden_states[-1].view(-1, self.hidden_size)).view(bs, t, -1)
        if not predict_input_vq:
            logits = logits[:,:,:self.num_vq]
        return logits
    def sample(self, input_ids, attention_mask, if_categorial=False, input_embeds=None):
        if input_embeds is not None:
            x = input_embeds.clone()
            if input_ids is not None:
                x_ids = input_ids.clone()
        else:
            x = input_ids.clone()
        xs = []
        assert input_ids.shape[0] == 1
        for k in range(x.shape[1]):
            if input_ids[0,k].item() < self.text_vocab_size or input_ids[0,k].item() >= self.text_vocab_size+self.num_vq:
                continue
            if input_embeds is not None:
                logits = self.forward(input_ids=x_ids[:,:k] if input_ids is not None else None, input_embeds=x[:,:k,:], attention_mask=attention_mask[:,:k])
            else:
                logits = self.forward(x[:,:k], attention_mask[:,:k])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                idx = idx.unsqueeze(-1)
            elif self.top_p is not None:
                sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumulative_probs < self.top_p
                nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1]+(1,)), nucleus[..., :-1]], dim=-1)
                sorted_probs[~nucleus] = 0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                dist = Categorical(sorted_probs.float())
                idx = dist.sample()
                idx = indices.gather(-1, idx.unsqueeze(-1))
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            if input_embeds is not None:
                x[:,k] = self.gpt.transformer.wte(idx.view(-1, 1)+self.text_vocab_size)
                if input_ids is not None:
                    x_ids[:,k] = idx.view(-1)+self.text_vocab_size
            else:
                x[:,k] = idx.view(-1)+self.text_vocab_size
            xs.append(idx.view(-1, 1))
        xs = torch.cat(xs, dim=1)
        return xs

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, extra_inputs=None):
        B, T, C = x.size() 

        if extra_inputs is not None:
            x = torch.cat((extra_inputs, x), dim=1)
            assert extra_inputs.shape[1] % T == 0, str(extra_inputs.shape[1])+", "+str(T)
            mask = self.mask[:,:,:T,:T].repeat(1, 1, extra_inputs.shape[1] // T + 1, extra_inputs.shape[1] // T + 1)
            T *= (1+extra_inputs.shape[1] // T)
        else:
            mask = self.mask
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if extra_inputs is not None:
            return y[:,:extra_inputs.shape[1],:].contiguous(), y[:,extra_inputs.shape[1]:,:].contiguous()
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        if isinstance(x, tuple):
            x, y = x
            resid_x, resid_y = self.attn(self.ln1(x), extra_inputs=self.ln1(y))
            x = x + resid_x
            x = x + self.mlp(self.ln2(x))
            y = y + resid_y
            y = y + self.mlp(self.ln2(y))
            return x, y
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

## NOTE: (EV) CARRY OVER CROSS ATTENTION STUFF
class CrossNorm(nn.Module):
  """ Norm Layer """

  def __init__(self, fn, size):
    super().__init__()
    self.norm = nn.LayerNorm(size, eps=1e-5)
    self.fn = fn

  def forward(self, x_data):
    x_norm = self.fn({'x_a':x_data['x_a'], 'x_b':self.norm(x_data['x_b'])})
    return x_norm

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    x: float Tensor to perform activation.
    Returns:
    `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + torch.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf

class CrossMLP(nn.Module):
  """ MLP Layer """

  def __init__(self, in_dim, out_dim, hidden_dim):
    super().__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    # self.activation = get_activation("gelu")
    self.activation = gelu
    self.l2 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x_data):
    out = self.l2(self.activation(self.l1(x_data['x_b'])))
    return {'x_a':x_data['x_a'], 'x_b':out}

class CrossResidual(nn.Module):
  """ Residual Layer """

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x_data):
    x_resid = self.fn(x_data)['x_b']
    return {'x_a':x_data['x_a'], 'x_b':x_resid+x_data['x_b']}

class CrossModalAttention(nn.Module):
    """ Cross Modal Attention Layer
    Given 2 modalities (a, b), computes the K,V from modality b and Q from
    modality a.
    """

    def __init__(self, in_dim, dim, heads=8, in_dim2=None, reverse_cross_modal=True):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5

        if in_dim2 is not None:
            self.to_kv = nn.Linear(in_dim2, in_dim2 * 2, bias=False)
        else:
            self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
        self.to_q = nn.Linear(in_dim, dim, bias=False)
        if in_dim2 is not None:
            dim2 = int((in_dim + in_dim2*2) / 3)
        else:
            dim2 = dim
        self.to_out = nn.Linear(dim2, dim)

        self.rearrange_qkv = Rearrange(
            "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        self.rearrange_q = Rearrange(
            "b n (h d) -> b h n d", h=self.heads)
        self.rearrange_kv = Rearrange(
            "b n (kv h d) -> kv b h n d", kv=2, h=self.heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")
        self.reverse_cross_modal = reverse_cross_modal

    def forward(self, x_data):
        x_a = x_data['x_a']
        x_b = x_data['x_b']

        if self.reverse_cross_modal:
            kv = self.to_kv(x_a)
            q = self.to_q(x_b)
        else:
            kv = self.to_kv(x_b)
            q = self.to_q(x_a)

        # q = x_a[:,:40,:].contiguous()
        # print(q.shape, kv.shape)
        """qkv = torch.cat((q, kv), dim=-1)
        qkv = self.rearrange_qkv(qkv)
        # print(qkv.shape)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]"""
        q = self.rearrange_q(q)
        kv = self.rearrange_kv(kv)
        k = kv[0]
        v = kv[1]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(dots, dim=-1)

        # print('a', x_a.shape)
        # print('b', x_b.shape)
        # print('attn', 'v', attn.shape, v.shape)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        # out = torch.einsum("bhij,bhjd->bhid", attn.permute(0, 1, 3, 2), q)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        # print('out', out.shape)
        return {'x_a':x_a, 'x_b':out}

class CrossTransformer(nn.Module):
    """ Transformer class
    Parameters
    ----------
    cross_modal : bool
    if true, uses cross-modal attention layers, else is the vanilla Transformer
    in_dim2 : int
    specifies the feature size of the second modality if using cross_modal
    """

    def __init__(self,
                in_size=50,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                in_dim2=None):
        super().__init__()
        blocks = []
        for i in range(num_hidden_layers):
            blocks.extend([
                CrossResidual(CrossNorm(CrossModalAttention(in_size, hidden_size,
                                                    heads=num_attention_heads,
                                                    in_dim2=in_dim2, reverse_cross_modal=True), hidden_size)),
                CrossResidual(CrossNorm(CrossMLP(hidden_size, hidden_size, intermediate_size),
                                    hidden_size))
            ])
        self.net = torch.nn.Sequential(*blocks)

    def forward(self, x_data):
        assert type(x_data) is dict
        x_data = self.net(x_data)
        # print('xa', x_data['x_a'].shape, 'xb', x_data['x_b'].shape, '-'*50)
        x = x_data['x_b']
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                extra_dim=[],
                gpt2_config=None):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        # self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.use_gpt2 = (gpt2_config is not None)
        self.gpt2_config = gpt2_config
        # transformer block
        if self.use_gpt2:
            self.blocks = nn.Sequential(*[GPT2Block(self.gpt2_config, layer_idx=i) for i in range(num_layers)])
        else:
            self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(math.ceil(block_size*1.6), embed_dim, 0.0, False)
        if len(extra_dim) > 0:
            self.extra_proj = nn.ModuleList([nn.Linear(dim, embed_dim) for dim in extra_dim])
            if len(extra_dim) == 2:
                self.cross_transformer = CrossTransformer(in_size=embed_dim, hidden_size=embed_dim, num_hidden_layers=12,
                    num_attention_heads=8, intermediate_size=3072, in_dim2=embed_dim)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, extra_inputs=None):
        # speaker_inp = None
        # audio_inp = None
        # # unwrap extra inputs
        # if len(extra_inputs) == 2:
        #     speaker_inp, audio_inp = extra_inputs
        # elif len(extra_inputs) == 1 and extra_inp[0].shape[-1] == 56:
        #     speaker_inp = extra_inputs
        # elif len(extra_inputs) == 1 and extra_inp[0].shape[-1] == 1024:
        #     audio_inp = extra_inputs

        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            clip_features_embedded = self.cond_emb(clip_feature)
            if len(clip_features_embedded.shape) == 2:
                clip_features_embedded = clip_features_embedded.unsqueeze(1)
            token_embeddings = torch.cat([clip_features_embedded, token_embeddings], dim=1)
           
        x = self.pos_embed(token_embeddings)
        y = None
        ymask = None
        # print(x.shape)
        mask = torch.tril(torch.ones((x.shape[1], x.shape[1]))).view(1, 1, x.shape[1], x.shape[1]).to(x.device)
        if len(idx) > 0 and extra_inputs is not None and len(extra_inputs) > 0:
            # assert t+sum([inp.shape[1] for inp in extra_inputs]) <= (1+len(self.extra_proj))*self.block_size, "Cannot forward, model block size "+str(self.block_size)+"is exhausted. Sequence_length "+str(t+extra_inputs.shape[1])
            
            if len(extra_inputs) == 2:
                # speaker_inp = self.pos_embed(self.extra_proj[0](speaker_inp))
                # audio_inp = self.pos_embed(self.extra_proj[1](audio_inp))
                for i, inp in enumerate(extra_inputs):
                    # print("in", inp[0].shape)
                    output = self.extra_proj[i](inp[0])
                    # print("out", output.shape)
                    # print("-"*100)
                    # assert len(inp) == 1
                extra_embeddings = [self.pos_embed(self.extra_proj[i](inp[0])) for i, inp in enumerate(extra_inputs)]
                extra_embeddings = [self.cross_transformer({'x_a': extra_embeddings[0], 'x_b': extra_embeddings[1]})]
            elif len(extra_inputs) == 1:
                extra_embeddings = [self.pos_embed(self.extra_proj[i](inp[0])) for i, inp in enumerate(extra_inputs)]
            y = torch.cat(extra_embeddings, dim=1)
            # print(y.shape, '-'*100) #4, 61, 1024
            if self.use_gpt2:
                x = self.blocks(hidden_states=torch.cat((x, y), dim=1), attention_mask=mask.repeat(1, 1, y.shape[1] // x.shape[1] + 1, y.shape[1] // x.shape[1] + 1))
                x = x[:,:x.shape[1],:]
            else:
                x = self.blocks((x, y))
                x = x[0]
        else:
            if self.use_gpt2:
                x = self.blocks(hidden_states=x, attention_mask=mask)
            else:
                x = self.blocks(x)
        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

