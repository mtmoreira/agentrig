"""ComfyUI local image integration."""

from agentrig.integrations.comfyui.images import (
    COMFYUI_IMAGE_CAPABILITY,
    COMFYUI_IMAGE_MAX_INPUTS,
    ComfyUIImageClient,
    ComfyUIImageClientFactory,
    ComfyUIImageGenerator,
    ComfyUIImageOperation,
    ComfyUIImageRequest,
    ComfyUIImageResult,
    ComfyUIImageSource,
)
from agentrig.integrations.comfyui.http import (
    ComfyUIHttpClientFactory,
    ComfyUIWorkflowCatalog,
    ComfyUIWorkflowTemplate,
)

__all__ = (
    "COMFYUI_IMAGE_CAPABILITY",
    "COMFYUI_IMAGE_MAX_INPUTS",
    "ComfyUIImageClient",
    "ComfyUIImageClientFactory",
    "ComfyUIImageGenerator",
    "ComfyUIImageOperation",
    "ComfyUIImageRequest",
    "ComfyUIImageResult",
    "ComfyUIImageSource",
    "ComfyUIHttpClientFactory",
    "ComfyUIWorkflowCatalog",
    "ComfyUIWorkflowTemplate",
)
