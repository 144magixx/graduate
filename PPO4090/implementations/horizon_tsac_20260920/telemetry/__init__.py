"""Independent CPU-only telemetry. Importing this package never starts training."""
from .recorder import Recorder, RecorderError

__all__ = ["Recorder", "RecorderError"]
