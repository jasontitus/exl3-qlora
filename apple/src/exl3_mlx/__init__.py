"""Apple silicon EXL3 primitives and project-based supervised fine-tuning."""

__all__ = ["EXL3Linear", "EXL3QLoRALinear", "reconstruct", "SFTConfig", "SFTTrainer"]


def __getattr__(name):
    # Keep CLI help, the project API, and Studio startup independent of Metal.
    if name in {"SFTConfig", "SFTTrainer"}:
        from . import sft

        return getattr(sft, name)
    if name in {"EXL3Linear", "EXL3QLoRALinear"}:
        from . import linear

        return getattr(linear, name)
    if name == "reconstruct":
        from .trellis import reconstruct

        return reconstruct
    raise AttributeError(name)
