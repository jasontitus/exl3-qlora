import collect

collect.SELECTION["mlx"] = [
    "d56ed24",
    "2d27ab0",
    "c7ff35d",
    "548dd80",
    "9008944",
    "a2a09fd",
]
original = collect.save


def save(path, obj):
    if path.name.endswith("-capture-progress.json"):
        path = path.with_name("performance-" + path.name)
    original(path, obj)


collect.save = save
records = collect.one_repo("mlx")
original(collect.ROOT / "inventory-performance.json", {"records": records})
