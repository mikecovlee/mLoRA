from typing import List, Tuple

import torch

from .abstracts import LLMFeedForward, LLMSparseMoe
from .config import LLMModelConfig, LoraMoeConfig, MixLoraConfig
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
        linear._moe_gates[adapter_config.adapter_name] = torch.nn.Linear(
            llm_config.dim_,
            adapter_config.num_experts_,
            bias=False,
            device=llm_config.device_,
            dtype=torch.float32,
        )
        linear.selective_hook_[adapter_config.adapter_name] = LoraMoe.selective_hook

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor) -> Tuple:
        return mlp._selective_forward(hidden_states, self.adapter_name_, moe_layer=self)


router_loss_dict = {"mixtral": MixtralRouterLoss, "switch": SwitchRouterLoss}


def router_loss_factory(config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    if config.router_loss_:
        return router_loss_dict[config.routing_strategy_](config)
    else:
        return None


moe_layer_dict = {
    "mixtral": MixtralSparseMoe,
    "switch": SwitchSparseMoe,
    "loramoe": LoraMoe,
}


def moe_layer_factory(args: LLMModelConfig, config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](args, config)
