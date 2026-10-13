"""CPU-only resource profiles; cache limits are part of the process budget."""

import math

PROFILES = ("low-memory", "development")
RESOURCE_FIELDS = (
    "profile",
    "memory_gib",
    "reserve_gib",
    "packed_cache_mib",
    "decoded_cache_mib",
    "prefix_cache_mib",
    "prefix_disk_cache",
    "prefix_disk_cache_gib",
    "allocator_cache_mib",
)


def resolve_resources(profile="low-memory", *, default_memory_gib=16.0, **overrides):
    if profile not in PROFILES:
        raise ValueError(f"Unknown resource profile: {profile}")
    unknown = set(overrides) - set(RESOURCE_FIELDS[1:])
    if unknown:
        raise ValueError(f"Unknown resource settings: {sorted(unknown)}")
    values = dict(
        profile=profile,
        memory_gib=default_memory_gib,
        reserve_gib=32.0,
        packed_cache_mib=128,
        decoded_cache_mib=0,
        prefix_cache_mib=0,
        prefix_disk_cache=None,
        prefix_disk_cache_gib=0.0,
        allocator_cache_mib=64,
    )
    if profile == "development":
        values.update(
            memory_gib=80.0, packed_cache_mib=8192, decoded_cache_mib=0, allocator_cache_mib=256
        )
    values.update({k: v for k, v in overrides.items() if v is not None})
    path, quota = values["prefix_disk_cache"], values["prefix_disk_cache_gib"]
    if (
        type(quota) not in (int, float)
        or not 0 <= quota <= 1024
        or not math.isfinite(quota)
        or (path is None and quota != 0)
        or (path is not None and (not isinstance(path, str) or not path.strip() or quota <= 0))
    ):
        raise ValueError(
            "prefix_disk_cache path and a finite positive quota up to 1024 GiB are required together"
        )
    validate_memory_budget(
        values["memory_gib"], values["allocator_cache_mib"], values["reserve_gib"]
    )
    for key in ("packed_cache_mib", "decoded_cache_mib", "prefix_cache_mib"):
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer MiB limit")
    cache_mib = sum(
        values[k]
        for k in (
            "packed_cache_mib",
            "decoded_cache_mib",
            "prefix_cache_mib",
            "allocator_cache_mib",
        )
    )
    workspace_mib = min(8192, values["memory_gib"] * 1024 / 2)
    if cache_mib > values["memory_gib"] * 1024 - workspace_mib:
        raise ValueError(
            "Combined packed, decoded, prefix and allocator caches must leave at least "
            f"{workspace_mib / 1024:g} GiB for model/workspaces inside the process budget; "
            "reduce caches or raise the budget (subject to measured system headroom)"
        )
    return values


def validate_memory_budget(memory_gib, allocator_cache_mib, reserve_gib):
    for key, value in (("memory_gib", memory_gib), ("reserve_gib", reserve_gib)):
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{key} must be finite")
    if not 1 <= memory_gib <= 1024 or reserve_gib < 0:
        raise ValueError("Use 1..1024 GiB process budget and nonnegative system reserve")
    if type(allocator_cache_mib) is not int or not 0 <= allocator_cache_mib <= 1024:
        raise ValueError("allocator_cache_mib must be an integer in 0..1024 MiB")


def add_resource_arguments(parser, *, optional_profile=False):
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=None if optional_profile else "low-memory",
        help="development: 80 GiB process, 32 GiB reserve, 8 GiB packed cache; decoded/prefix caches off",
    )
    parser.add_argument("--memory-gib", type=float)
    parser.add_argument("--reserve-gib", type=float)
    parser.add_argument("--packed-cache-mib", "--cache-mib", type=int, dest="packed_cache_mib")
    parser.add_argument("--decoded-cache-mib", type=int)
    parser.add_argument(
        "--prefix-cache-mib",
        type=int,
        help="Frozen-prefix activation cache inside the process budget, MiB (default 0/off)",
    )
    parser.add_argument(
        "--prefix-disk-cache", help="Opt-in exact frozen-prefix disk cache directory"
    )
    parser.add_argument(
        "--prefix-disk-cache-gib", type=float, help="Disk cache quota (requires path)"
    )
    parser.add_argument("--allocator-cache-mib", type=int)


def resource_overrides(args):
    return {
        key: getattr(args, key) for key in RESOURCE_FIELDS if getattr(args, key, None) is not None
    }


def resources_from_args(args, *, default_memory_gib=16.0):
    overrides = resource_overrides(args)
    if "packed_cache_mib" not in overrides and getattr(args, "cache_mib", None) is not None:
        overrides["packed_cache_mib"] = args.cache_mib
    return resolve_resources(default_memory_gib=default_memory_gib, **overrides)
