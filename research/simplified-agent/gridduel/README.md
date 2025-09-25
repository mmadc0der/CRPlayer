# GridDuel RL Playground

Minimal, reproducible RL research playground for a two-player Grid Duel game. Includes a Gymnasium-compatible environment, baseline policies, a CLI to run head-to-head matches, a parallel runner for evaluation/training loops, an interactive notebook, examples, and tests.

## Features

- Gymnasium-compatible environment `GridDuelEnv`
- Policies: Random, Rule-based, Human (CLI), Torch model policy (optional)
- CLI to run matches and save replays
- Parallel runner for many matches in parallel
- Interactive notebook with ipywidgets
- Tests with pytest

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Smoke test: random vs random
python -m cli.gridduel_cli --player0 random --player1 random --episodes 3

# Run tests
pytest -q
```

Note: Run the commands from inside the `gridduel/` directory.

## Torch (optional)

`TorchModelPolicy` requires PyTorch. Install separately if needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# or a CUDA build per your system
```

## CLI Examples

```bash
# human vs random (ASCII render)
python -m cli.gridduel_cli --player0 human --player1 random --episodes 1 --render ascii

# random vs model
python -m cli.gridduel_cli --player0 random --player1 model:models/agent.pt --episodes 50 --parallel 4 --record results.json
```

## API

- `gridduel_env.env.GridDuelEnv`: core multi-agent environment with `step_both(a0, a1)`
- `gridduel_env.wrappers.GridDuelEnvGym`: single-agent Gym wrapper vs fixed opponent
- `gridduel_env.policies.*`: policies with a simple `act(obs)` API
- `runner.parallel_runner.run_matches`: run many matches in parallel

## Layout

```
gridduel/
├── README.md
├── requirements.txt
├── gridduel_env/
│   ├── __init__.py
│   ├── env.py
│   ├── wrappers.py
│   ├── policies.py
│   └── utils.py
├── cli/
│   └── gridduel_cli.py
├── runner/
│   ├── parallel_runner.py
│   └── evaluator.py
├── notebooks/
│   └── interactive.ipynb
├── examples/
│   ├── train_example.py
│   └── play_example.py
└── tests/
    ├── test_env.py
    ├── test_policies.py
    └── test_parallel.py
```

## Development

- Python 3.10+
- Deterministic seeding supported via `seed` in env and runner
- No heavy infra required; optional PyTorch for TorchModelPolicy

## License

MIT