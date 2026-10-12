"""Validate all three matrices of expert 0 from the pinned 2.8-bpw target."""

import json
from pathlib import Path

from exl3_mlx.checkpoint import Checkpoint
from exl3_mlx.moe import DeepSeekExpert
from exl3_mlx.validation import validate_expert

root = Path(__file__).resolve().parents[1]
bases = [
    Checkpoint(root / "artifacts" / f"deepseek-v41-expert0-w{i}.safetensors").load_linear(
        f"layers.0.ffn.experts.0.w{i}"
    )
    for i in (1, 2, 3)
]
report = validate_expert(DeepSeekExpert(*bases), bases)
report.update(
    repo="jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw",
    revision="915e1d0de8ee0a99944c5bc88ab802a639c63380",
    key="layers.0.ffn.experts.0",
)
(root / "reports/deepseek-v41-complete-expert.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
