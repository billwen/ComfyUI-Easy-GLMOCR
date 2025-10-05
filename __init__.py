from .nodes import *
from typing_extensions import override

class DotsOCRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DownloadAndLoadDotsOCRModel,
            ApplyDotsOCR
        ]

async def comfy_entrypoint() -> DotsOCRExtension:
    return DotsOCRExtension()