import torch

from .config import LLMModelConfig, MixConfig
from .mix_lora import (
    MixtralRouterLoss,
    MixtralSparseMoe,
    SwitchRouterLoss,
    SwitchSparseMoe,
)

router_loss_dict = {"mixtral": MixtralRouterLoss, "switch": SwitchRouterLoss}


def router_loss_factory(config: MixConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    if config.router_loss_:
        return router_loss_dict[config.routing_strategy_](config)
    else:
        return None


moe_layer_dict = {"mixtral": MixtralSparseMoe, "switch": SwitchSparseMoe}


def moe_layer_factory(args: LLMModelConfig, config: MixConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](args, config)
