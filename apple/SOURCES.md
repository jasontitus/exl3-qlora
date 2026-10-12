# Sources and reuse

| Project | Version inspected / used | Role |
| --- | --- | --- |
| [UnstableLlama/exl3-qlora](https://github.com/UnstableLlama/exl3-qlora) | baseline `068f586c00724e3eabbf4165fb66c355da9af4df` | This repository; EXL3 format, codebooks, permutations, transforms and QLoRA math |
| [MLX](https://github.com/ml-explore/mlx) | runtime 0.32.3, locked | Metal kernels, arrays, autodiff, optimizers |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | runtime 0.32.0, locked; source review `ba8a08fd2ed72bbc53de84f89933f91fe4743e4a` | Imported dataset processors, masked causal loss and LoRA forward; [MIT notice](configs/MLX_LM_LICENSE) |
| [DeepSeek V4.1 official inference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py) | `5254448103b6254fc646dacbca1cbbf20e9216c1` | Architecture reference and adapted math; [MIT notice](configs/DEEPSEEK_LICENSE) |
| [PonyExl3](https://github.com/beamivalice/PonyExl3) | inspected `8e7fa6b1556f59fc669e25087903b279b9b0346f` | Evaluated fused Metal EXL3 inference kernels; **no code incorporated** |

MLX-LM's [DeepSeek V4.1 model](https://github.com/ml-explore/mlx-lm/blob/ba8a08fd2ed72bbc53de84f89933f91fe4743e4a/mlx_lm/models/deepseek_v41.py)
explicitly rejects configurations with Engram layers. That is why this port retains
the local repack reader, Engram implementation and eager recomputation loop.
MLX-LM's compiled training loop assumes ordinary in-memory MLX models; disk reads,
routing decisions and materialized first-order VJPs here require an eager loop.

`uv.lock` records all installed dependency versions. Source checkouts under ignored
`artifacts/reference` are read-only references, not build dependencies. Model
weights and tokenizers are not redistributed by this package.
