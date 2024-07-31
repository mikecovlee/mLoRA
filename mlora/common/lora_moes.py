from typing import List, Tuple

import torch

from .config import LLMModelConfig, LoraMoeConfig, MixLoraConfig
from .mix_lora import (
    MixtralRouterLoss,
    MixtralSparseMoe,
    SwitchRouterLoss,
    SwitchSparseMoe,
)
from .model import LLMFeedForward, LLMSparseMoe


class LoraMoe(LLMSparseMoe):
    def __init__(self, llm_config: LLMModelConfig, config: LoraMoeConfig) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            llm_config.dim_,
            config.num_experts_,
            bias=False,
            device=llm_config.device_,
            dtype=self.dtype_,
        )
        self.experts_ = config.num_experts_
        self.blc_alpha_ = config.blc_alpha_
        self.blc_weight_ = config.blc_weight_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

    def forward(self, mlp: LLMFeedForward, hidden_states: torch.Tensor) -> Tuple:
        pass


router_loss_dict = {"mixtral": MixtralRouterLoss, "switch": SwitchRouterLoss}


def router_loss_factory(config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    if config.router_loss_:
        return router_loss_dict[config.routing_strategy_](config)
    else:
        return None


moe_layer_dict = {"mixtral": MixtralSparseMoe, "switch": SwitchSparseMoe}


def moe_layer_factory(args: LLMModelConfig, config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](args, config)
