import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.phi3.modeling_phi3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)
from mlora.backends import backend
from mlora.common import (
    CHECKPOINT_CLASSES,
    Cache,
    FeedForward,
    Linear,
    LLMAttention,
    LLMDecoder,
    LLMFeedForward,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    flash_attention_forward,
    prepare_4d_causal_attention_mask,
)
from mlora.common.mix_lora import _mixtral_slice_tensor
from mlora.utils import copy_parameters
from .modeling_gemma2 import Gemma2RotaryEmbedding


@dataclass
class Phi3Config(LLMModelConfig):
    rms_norm_eps_: float = 1e-6
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: dict = None
    use_sliding_window_: bool = False
    sliding_window_: int = 4096
    resid_pdrop: float = 0.0
    rms_norm_eps = 1e-5
    intermediate_size = 8192


class Phi3RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)

        print("Phi3 RMSNorm passed.")
        return (self.weight_ * data).to(input_dtype)


class Phi3Embedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)
        # normalizer = torch.tensor(self.normalizer_, dtype=data.dtype)
        print("Phi3Embedding passed.")
        return data


class Phi3Attention(LLMAttention):
    def __init__(
        self, qkv_proj: nn.Module, o_proj: nn.Module, layer_idx: int, args: Phi3Config
    ) -> None:
        super().__init__()
        # attention
        self.qkv_proj_ = Linear(qkv_proj, args.device_)
        self.o_proj_ = Linear(o_proj, args.device_)
        # config
        self.layer_idx = layer_idx
        self.args_ = args
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.max_position_embeddings = args.max_position_embeddings
        self.original_max_embeddings = args.original_max_position_embeddings
        self.rope_theta_ = args.rope_theta
        self.head_dim_ = self.dim_ // self.n_heads_
        self.dtype_ = args.dtype_
        self.is_causal_ = True
        self.sliding_window_ = (
            args.sliding_window_
            if args.use_sliding_window_ and not bool(layer_idx % 2)
            else None
        )

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "qkv_proj": self.qkv_proj_,
            "o_proj": self.o_proj_,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj_.forward(hidden_states, input_args)
        query_pos = self.n_heads_ * self.head_dim_
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.n_kv_heads_ * self.head_dim_]
        value_states = qkv[..., query_pos + self.n_kv_heads_ * self.head_dim_ :]

        query_states = query_states.view(
            bsz, q_len, self.n_heads_, self.head_dim_
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        # apply rotary embedding
        cos, sin = rotary_emb
        assert query_states.dtype == key_states.dtype
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window_,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx_, cache_kwargs
            )

        value_states = repeat_kv(value_states, self.n_rep_)
        key_states = repeat_kv(key_states, self.n_rep_)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim_)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        print("Phi3 Attention passed.")

        return self.o_proj_(attn_output, input_args)


class Phi3FlashAttention2(Phi3Attention):
    """
    Phi-3 flash attention module. This module inherits from `Phi3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(
        self,
        qkv_proj: nn.Module,
        o_proj: nn.Module,
        layer_idx: int,
        args: Phi3Config,
    ) -> None:
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(qkv_proj, o_proj, layer_idx, args)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):

        bsz, q_len, _ = hidden_states.size()

        # cutting
        qkv = self.qkv_proj_.forward(hidden_states, input_args)
        query_pos = self.n_heads_ * self.head_dim_
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.n_kv_heads_ * self.head_dim_]
        value_states = qkv[..., query_pos + self.n_kv_heads_ * self.head_dim_ :]

        # viewing
        query_states = query_states.view(
            bsz, q_len, self.n_heads_, self.head_dim_
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        # sin & cos
        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Activate slicing cache
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window_,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx_, cache_kwargs
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if backend.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            is_causal=self.is_causal_,
            sliding_window=getattr(self.args_, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        ).to(input_dtype)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj_(attn_output, input_args)

        return attn_output


PHI3_ATTENTION_CLASSES = {
    "eager": Phi3Attention,
    "flash_attn": Phi3FlashAttention2,
}


class Phi3DecoderLayer(LLMDecoder):
    def __init__(self, layer_id: int, config: Phi3Config) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_: Phi3Attention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: Phi3RMSNorm = None
        self.post_attention_layernorm_: Phi3RMSNorm = None

        self.config_ = config
        self.is_sliding_ = not bool(layer_id % 2)
        self.post_feedforward_layernorm_: Phi3RMSNorm = None
        self.sliding_window_ = config.sliding_window_

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(
            config.dim_, eps=config.rms_norm_eps
        )

    def state_dict(self) -> Dict[str, nn.Module]:
        linear_layers = self.self_attn_.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm_(hidden_states)
        # Self Attention
        hidden_states = self.self_attn_.forward(
            hidden_states,
            input_args,
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states, router_logits = self.mlp_.forward(hidden_states, input_args)
        hidden_states = residual + hidden_states

        print("phi3 Decoder passed.")
        return hidden_states, *router_logits


class Phi3SequentialWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "Phi3Embedding":
            output = self.wrapper_module_.forward(input[0])
            if input[-1].gradient_checkpoint_ != "none":
                output = output.requires_grad_(True)
            return (output,) + input[1:]
        elif module_name == "Phi3RMSNorm":
            output = self.wrapper_module_.forward(input[0])
            return (output,) + input[1:]
        elif module_name == "Phi3DecoderLayer":
            outputs = CHECKPOINT_CLASSES[input[-1].gradient_checkpoint_](
                self.wrapper_module_.forward,
                *input,
            )
            if len(outputs) > 1:
                self.router_probs_ = outputs[1:]
            return (outputs[0],) + input[1:]
        else:
            raise f"module invalid: {module_name}"


class Phi3MLP(LLMFeedForward):
    def __init__(self, gate: nn.Module, down: nn.Module, args: Phi3Config) -> None:
        super().__init__()
        # feed forward
        self.gate_up_ = Linear(gate, args.device_)
        self.down_ = Linear(down, args.device_)
        self.act_ = ACT2FN[args.hidden_act_]

    def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "gate_up_proj": self.gate_up_,
            "down_proj": self.down_,
        }

    def _batch_forward(
        self, hidden_states: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        up_states = self.gate_up_.forward(hidden_states, input_args)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_(gate)

        print("Phi3MLP._batch_forward() passed.")
        return self.down_(up_states, input_args)

    def _lora_forward(
        self, lora_name: str, act_fn: nn.Module, data: torch.Tensor
    ) -> torch.Tensor:
        # Applying LoRA weights to FFN weights
        if lora_name in self.gate_up_.loras_:
            gate = self.gate_up_.loras_[lora_name].forward(
                self.gate_up_.base_layer_.forward(data), data
            )
        else:
            gate = self.gate_up_.base_layer_.forward(data)

        if lora_name in self.gate_up_.loras_:
            up = self.gate_up_.loras_[lora_name].forward(
                self.gate_up_.base_layer_.forward(data), data
            )
        else:
            up = self.gate_up_.base_layer_.forward(data)

        act_result = act_fn(gate) * up
        if lora_name in self.down_.loras_:
            print("Phi3MLP._lora_forward(1) passed.")
            return self.down_.loras_[lora_name].forward(
                self.down_.base_layer_.forward(act_result), act_result
            )
        else:
            print("Phi3MLP._lora_forward(2) passed.")
            return self.down_.base_layer_.forward(act_result)

    def _mixlora_forward(
        self, moe_name, act_fn, expert_mask, hidden_states, input_dtype
    ):
        common_gate_up = self.gate_up_.base_layer_.forward(hidden_states.to(input_dtype)).to(hidden_states.dtype)

        final_expert_states = []
        for expert_idx in range(expert_mask.shape[0]):
            _, top_x = torch.where(expert_mask[expert_idx])

            lora_name = f"moe.{moe_name}.experts.{expert_idx}"
            if lora_name in self.gate_up_.loras_:
                lora_data = _mixtral_slice_tensor(hidden_states, top_x, input_dtype)
                gate_up_states = self.gate_up_.loras_[lora_name].forward(
                    _mixtral_slice_tensor(common_gate_up, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                gate_up_states = _mixtral_slice_tensor(common_gate_up, top_x, input_dtype)

            gate, up_states = gate_up_states.chunk(2, dim=-1)
            act_result = up_states * self.act_(gate)

            if lora_name in self.down_.loras_:
                final_expert_states.append(
                    self.down_.loras_[lora_name].forward(  # LoRA a,b
                        self.down_.base_layer_.forward(act_result),
                        act_result,
                    )
                )
            else:
                final_expert_states.append(self.down_.base_layer_.forward(act_result))

        print("Phi3MLP._mixlora_forward() passed.")
        return final_expert_states


class Phi3OutputLayer(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head_(hidden_states)
        return logits


class Phi3ForCausalLM(LLMForCausalLM):
    # father class? transformer.PreTrainedModel?
    config_class = Phi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True
    _tied_weights_keys = ["lm_head.weight"]

    _version = "0.0.5"

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: Phi3Embedding = None
        self.norm_: Phi3Embedding = None
        self.rotary_emb_ = Gemma2RotaryEmbedding(  # 此部分等效于 init_rope()
            # args.head_dim_,  # here should be 96 (3072/32)
            config.dim_ // config.n_heads_,
            max_position_embeddings=config.max_seq_len_,
            base=config.rope_theta_,
            device=config.device_,
        )
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.layers_: List[Phi3DecoderLayer] = []

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens_(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb_(input_tensor, position_ids)

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm_(hidden_states)

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update({"embedding": Phi3SequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: Phi3SequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update({"norm": Phi3SequentialWrapper(self.norm_)})
        seq_module.move_to_end("norm")

        return seq_module

    def causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
    ) -> torch.Tensor:

        return prepare_4d_causal_attention_mask(
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def cache_implementation(self) -> str:
        if self.config_.use_sliding_window_ and self.config_.sliding_window_:
            return "hybrid"
        else:
            return "dynamic"

    def model_config(self) -> Phi3Config:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = backend.default_device_name(),
    ):
        llm_config = llm_model.config
        llm_args = Phi3Config(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=llm_config.hidden_act,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            use_sliding_window_=use_sliding_window,
            sliding_window_=llm_config.sliding_window,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if use_sliding_window and attn_impl != "flash_attn":
            raise ValueError(
                f"Can not use sliding window attention with {attn_impl} attention."
            )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = Phi3ForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = Phi3Embedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = Phi3RMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        # copy_parameters(llm_model.model.embed_tokens, model.embed_tokens_.embed_tokens)
        # copy_parameters(llm_model.model.final_layernorm, model.final_layernorm_.layernorm_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = Phi3DecoderLayer(idx, llm_args)
            decoder.layer_id_ = idx
            decoder.self_attn_ = PHI3_ATTENTION_CLASSES[llm_args.attn_implementation_](
                layer.self_attn.qkv_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(
                Phi3MLP(
                    layer.mlp.gate_up_proj,
                    layer.mlp.down_proj,
                    llm_args,
                )
            )
            decoder.input_layernorm_ = Phi3RMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = Phi3RMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_
            )
            model.layers_.append(decoder)

        print("phi3FormPretrained passed ")
        return model