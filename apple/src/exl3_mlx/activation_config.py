"""CPU-only serialization rules for the experimental activation arithmetic mode."""

ACTIVATION_MODES = ("checkpoint", "bf16")


def validate_activation_mode(value):
    if not isinstance(value, str) or value not in ACTIVATION_MODES:
        raise ValueError("activation_mode must be 'checkpoint' or experimental 'bf16'")
    return value


def normalize_adapter_config(config):
    """Legacy absence means checkpoint mode; explicit null/unknown values fail."""
    result = dict(config)
    result["activation_mode"] = validate_activation_mode(
        result.get("activation_mode", "checkpoint")
    )
    return result


def activation_description(mode):
    mode = validate_activation_mode(mode)
    if mode == "checkpoint":
        return "fp8/fp4 activation roundtrips with identity STE"
    return (
        "experimental: differentiable-path fp8/fp4 activation roundtrips bypassed; "
        "FrozenIndexer quantizers, existing tensor dtypes, BF16 casts, packed weights and normal routing retained"
    )
