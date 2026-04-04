from .nodes import *
from typing_extensions import override

class GlmOCRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DownloadAndLoadGlmOCRModel,
            ApplyGlmOCR
        ]

async def comfy_entrypoint() -> GlmOCRExtension:
    return GlmOCRExtension()
