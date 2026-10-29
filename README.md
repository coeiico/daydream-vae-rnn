# daydream-vae-rnn

**World Models.** A VAE + MDN-RNN world model with a controller trained entirely inside its own dream.

Part of the **Latent Space Project**: 18 small PyTorch repos that test one idea, that neural networks encode concepts and strategy in their latent space, and that we can read and edit those representations. Everything trains on a single consumer GPU or a laptop.

| Roadmap | Phase | Depends on | Status |
| --- | --- | --- | --- |
| 17 of 18 | 5. Representation and generation | VAE code from knobs-vae | Not started (scaffolded) |

## Question

Can an agent learn a compressed model of its world and learn to act inside its own imagination?

## Plan

- **Architecture:** a convolutional VAE compresses frames into a latent vector; an MDN-RNN (an LSTM predicting a mixture of Gaussians) predicts the next latent; a tiny linear controller is trained with CMA-ES. Reuse VAE code from knobs-vae.
- **Environment:** start with a simple 64×64 pixel environment; move to gymnasium's CarRacing (the original paper's task) only once the pipeline works.

## Experiments

- Reconstructions.
- Decoded "dream" rollouts.
- A controller trained entirely in the dream, then tested in the real environment.

## Done when

- The dream-trained controller beats random play in the real environment, with a short dream video in the README.

Plus the definition of done every Latent Space Project repo shares:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Notes

- The most compute-heavy project, so it sits late in the roadmap.
- CarRacing needs gymnasium's Box2D extra (`uv add "gymnasium[box2d]"`); add it only when moving to that environment.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
daydream-vae-rnn/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/daydream_vae_rnn/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Ha and Schmidhuber 2018, World Models

## License

MIT
