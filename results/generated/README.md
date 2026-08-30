# Sample AI-generated kernels

A representative slice of what the codegen loop produced with `qwen3.8:27b`, the model we ship: the best correct kernel per target, plus one example of each failure mode that run produced (it produced one). Full per-attempt verdicts are in `../codegen.json`.

`../codegen_<model>_<arch>.json` holds one arm per model per GPU from the bake-off in `docs/CODEGEN.md`; `../codegen_repair3_sm_80.json` is the repair ablation; `../model_aliases.json` records which model id the gateway actually serves for each requested id.

None of these ship. They are proposals, and a public submission should not contain code no person has read. See `docs/CODEGEN.md`.

- `add_mask_layernorm_e259a16959.py` -- layernorm: BEST correct kernel, 2.46x vs torch, envelope 0.042
- `bias_gelu_ac7f63f920.py` -- gelu: BEST correct kernel, 3.91x vs torch, envelope 0.031
- `bias_gelu_0ea01c2425.py` -- gelu: example of numeric_fail
