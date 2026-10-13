import json
from pathlib import Path

import collect

collect.SELECTION = {
    "mlx": ["8295d5e", "77bf1fa", "789be76", "494c8a6", "4f8d916", "02adf7b", "a131614", "b08ec31"]
}
rows = collect.one_repo("mlx")
p = Path(__file__).resolve().parent / "inventory.json"
old = json.loads(p.read_text())
old["supplement_selection"] = collect.SELECTION
old["records"] += rows
collect.save(p, old)
