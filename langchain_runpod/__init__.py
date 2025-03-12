from importlib import metadata

from langchain_runpod.chat_models import ChatRunPod
from langchain_runpod.llms import RunPod

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatRunPod",
    "RunPod",
    "__version__",
]
