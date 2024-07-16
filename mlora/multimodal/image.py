from typing import Dict

import torch
from transformers.activations import ACT2FN

from mlora.common import LLMModelArgs, LLMMultiModalProjector, VisionConfig


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class ImageProjector(LLMMultiModalProjector):
    def __init__(self, model_config: LLMModelArgs, vision_config: VisionConfig) -> None:
        super().__init__()

        self.vision_proj_ = torch.nn.Linear(
            vision_config.dim_,
            model_config.dim_,
            bias=True,
            device=model_config.device_,
        )
        self.act_ = ACT2FN[vision_config.hidden_act_]
        self.text_proj_ = torch.nn.Linear(
            model_config.dim_, model_config.dim_, bias=True, device=model_config.device_
        )

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {
            "vision_proj": self.vision_proj_.weight,
            "text_proj": self.text_proj_.weight,
        }

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.vision_proj_(image_features)
        hidden_states = self.act_(hidden_states)
        hidden_states = self.text_proj_(hidden_states)
        return hidden_states


class ImageInputLayer:
    def __init__(self):
        pass
