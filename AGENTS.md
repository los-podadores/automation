# AGENTS.md

## Project Overview

University robotics project ("Los Podadores") implementing and comparing autonomous lawn mower coverage path planning algorithms. Two subsystems: RL-based (`src/v3/`) and classical algorithms (`src/algo/`).

## Setup

```bash
# Required: Python 3.12, Nix with direnv
direnv allow  # Activates Nix flake with SDL2, uv, ruff
uv sync       # Install Python dependencies
```

## Key Commands

```bash
# Training (16M timesteps, 20 parallel envs)
uv run python src/v3/train.py

# Resume training
uv run python src/v3/train.py --resume models/v3/ppo_v3_XXXXX.zip

# Visualization
uv run python src/v3/visualize.py --model models/v3/ppo_v3_final.zip --phase 1 --episodes 5

# Classical algorithms
uv run python src/algo/smart_mower.py
uv run python src/algo/flood_fill_with_map.py
uv run python src/algo/boustrophedon_path.py
uv run python src/algo/run_cstar.py --phase 1 --render human
uv run python src/algo/run_estar.py --phase 3 --episodes 5
uv run python src/algo/sstc/new.py
```

## Architecture

- `src/v3/` - RL approach (primary). Gymnasium env with PPO, curriculum learning (8 phases), custom CNN feature extractor.
- `src/algo/` - Classical algorithms (baselines). Each is self-contained with its own simulation loop.
- `src/v3/env/` - Decomposed environment sub-package (config, maps, sensors, collision, transforms, renderer).
- `ppo_v3_final.zip` - Pre-trained model checkpoint (committed to repo).

## Conventions

- **Package manager**: `uv` (not pip/poetry)
- **Linter/formatter**: `ruff` (available via Nix, no pyproject.toml config)
- **Python**: 3.12 required (`.python-version`)
- **Entry points**: Scripts run from project root with `uv run python src/...`
- **No tests**: No test suite exists despite testing skills being configured.
- **Spanish comments**: Some code comments are in Spanish (university project context).

## Gotchas

- `src/v3/train.py` imports from `architectures`, `robot_env` directly (not relative imports) - must run from `src/v3/` or adjust sys.path.
- Classical algo files in `src/algo/` are standalone scripts with `if __name__ == "__main__"` blocks.
- Pygame visualization requires SDL2 libraries (provided by Nix flake).
- `models/` and `logs/` directories are gitignored - created during training.
