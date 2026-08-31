# scripts/

Everything here runs on any CUDA GPU, with no cluster and no API key required.
Each degrades to something correct if a dependency is missing, and says so.

| script | what it does |
|---|---|
| `run_official.py` | the organizer's benchmark script, unmodified, with our layer substituted in. The thing to run first. |
| `showcase.py` | the guided walkthrough — seven acts (`--list` shows them; the default runs the first five in ~3 min). `--no-gpu` runs the three that read committed artifacts, so it works with no GPU at all. |
| `shape14.py` | official shape 14, which the reference implementation cannot run. `--scan` sweeps sequence length to find your GPU's limit. |
| `usecase.py` | the ranking workload, measured end to end. `--cold` shows what an untuned system does; `--tune` fixes that. |
| `report.py` | regenerates `docs/RESULTS.md` from `results/*.json`. |
| `dashboard.py` | regenerates `docs/dashboard.html`. Self-contained; no network. |
| `impact.py` | recomputes the throughput and energy figures from committed artifacts. |
| `pick_model.py` | scores every model your LLM key can reach on proposal quality. **Read `docs/CODEGEN.md` before trusting its recommendation** — we measured it to be anti-predictive of the thing that matters. |

## What is not here

The Slurm job files and the SSH driver that produced our cluster numbers are not
in the repository. They only run on one specific university cluster, so
they are provenance rather than instructions, and every result they produced is
committed as JSON in `results/`.

Each has a portable equivalent that does the same work on your hardware:

| we ran | you run |
|---|---|
| the official-shape tuning job | `python -m kernelforge.cli tune --shapes-file official_shapes.txt` |
| the verification job | `python -m kernelforge.cli verify --shapes-file official_shapes.txt --demote` |
| the full cross-architecture sweep | `python -m kernelforge.cli sweep` |
| the proposer head-to-head | `cli agent --provider heuristic --tag heuristic`, then `--provider llm --tag llm` |
| the kernel codegen and model bake-off | `python -m kernelforge.cli codegen` |
| shape 14, and its full-batch correctness gate | `python scripts/shape14.py`, `python scripts/shape14.py --gate --batch 32` |
| the untuned-fallback portability check | `cli verify --shapes-file official_shapes.txt --untuned --json results/portability_<arch>.json` |

Credentials are never stored in this repository and never passed on a command
line; see `.env.example`.
