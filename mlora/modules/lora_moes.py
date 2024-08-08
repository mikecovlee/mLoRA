import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .abstracts import LLMFeedForward, LLMSparseMoe
from .config import LLMModelConfig, LoraMoeConfig, MixLoraConfig, MolaConfig
from .lora_linear import Linear
from .mix_lora import (
    MixtralRouterLoss,
    MixtralSparseMoe,
    SwitchRouterLoss,
    SwitchSparseMoe,
)


class LoraMoe(LLMSparseMoe):
    def __init__(self, llm_config: LLMModelConfig, config: LoraMoeConfig) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.experts_ = config.num_experts_
        # blc loss is not available due to lack of generality
        # TODO: add blc loss support
        self.blc_alpha_ = config.blc_alpha_
        self.blc_weight_ = config.blc_weight_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

    @staticmethod
    def selective_hook(
        linear: Linear,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        moe_layer: "LoraMoe",
    ):
        lora_route: torch.nn.Linear = linear._loramoe_gates[moe_layer.adapter_name_]
        route_weight = torch.nn.functional.softmax(
            lora_route(hidden_states), dim=-1, dtype=torch.float32
        ).to(hidden_states.dtype)
        for expert_idx in moe_layer.experts_:
            lora = linear.loras_[f"moe.{moe_layer.adapter_name}.experts.{expert_idx}"]
            residual = residual + torch.unsqueeze(
                route_weight[:, :, expert_idx], -1
            ) * (
                lora.lora_b_(
                    lora.lora_a_(lora.dropout_(hidden_states.to(torch.float32)))
                )
                * lora.scaling_
            )

        return residual

    @staticmethod
    def adapter_initializer(
        llm_config: LLMModelConfig,
        adapter_config: LoraMoeConfig,
        linear: Linear,
    ):
        if not hasattr(linear, "_moe_gates"):
            linear._moe_gates = {}
        lora_route = torch.nn.Linear(
            llm_config.dim_,
            adapter_config.num_experts_,
            bias=False,
            device=llm_config.device_,
            dtype=torch.float32,
        )
        torch.nn.init.kaiming_uniform_(
            lora_route.weight, a=math.sqrt(adapter_config.router_init_range_)
        )
        linear._moe_gates[adapter_config.adapter_name] = lora_route
        linear.selective_hook_[adapter_config.adapter_name] = LoraMoe.selective_hook

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor) -> Tuple:
        return (
            mlp._selective_forward(hidden_states, self.adapter_name_, moe_layer=self),
            None,
        )


class MolaSparseMoe(LLMSparseMoe):
    def __init__(self, llm_config: LLMModelConfig, config: MolaConfig) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.experts_ = config.num_experts_
        self.topk_ = config.top_k_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

    @staticmethod
    def selective_hook(
        linear: Linear,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        moe_layer: "MolaSparseMoe",
    ):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(moe_layer.dtype_)
        router_logits = linear._loramoe_gates[moe_layer.adapter_name_](hidden_states)
        routing_weights_before = F.softmax(router_logits, dim=1, dtype=moe_layer.dtype_)
        routing_weights, selected_experts = torch.topk(
            routing_weights_before, moe_layer.topk_, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=moe_layer.experts_
        ).permute(2, 1, 0)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=moe_layer.dtype_,
            device=hidden_states.device,
        )

        for expert_idx in moe_layer.experts_:
            lora = linear.loras_[f"moe.{moe_layer.adapter_name}.experts.{expert_idx}"]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            expert_output = (
                lora.lora_b_(
                    lora.lora_a_(lora.dropout_(current_state.to(torch.float32)))
                )
                * lora.scaling_
            )
            current_hidden_states = (
                expert_output * routing_weights[top_x_list, idx_list, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(moe_layer.dtype_)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(input_dtype)

        return residual + final_hidden_states

    @staticmethod
    def adapter_initializer(
        llm_config: LLMModelConfig,
        adapter_config: MolaConfig,
        linear: Linear,
    ):
        if not hasattr(linear, "_moe_gates"):
            linear._moe_gates = {}
        lora_route = torch.nn.Linear(
            llm_config.dim_,
            adapter_config.num_experts_,
            bias=False,
            device=llm_config.device_,
            dtype=torch.float32,
        )
        torch.nn.init.kaiming_uniform_(
            lora_route.weight, a=math.sqrt(adapter_config.router_init_range_)
        )
        linear._moe_gates[adapter_config.adapter_name] = lora_route
        linear.selective_hook_[adapter_config.adapter_name] = LoraMoe.selective_hook

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor) -> Tuple:
        return (
            mlp._selective_forward(hidden_states, self.adapter_name_, moe_layer=self),
            None,
        )


router_loss_dict = {"mixlora": MixtralRouterLoss, "mixlora-switch": SwitchRouterLoss}


def router_loss_factory(config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    if config.router_loss_:
        return router_loss_dict[config.routing_strategy_](config)
    else:
        return None


moe_layer_dict = {
    "mixlora": MixtralSparseMoe,
    "mixlora-switch": SwitchSparseMoe,
    "loramoe": LoraMoe,
    "mola": MolaSparseMoe,
}


def moe_layer_factory(args: LLMModelConfig, config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in moe_layer_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](args, config)
