# Contributing to RKE

Thank you for your interest in contributing to the Recursive Knowledge Engine!

## Development Setup

```bash
git clone https://github.com/DRCubix/RKEmemory.git
cd RKEmemory
bash scripts/setup.sh
source .venv/bin/activate
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check src/
ruff format src/
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Write tests for new functionality
3. Ensure `rke status` works with your changes
4. Open a PR with a clear description of changes

## Architecture Decisions

Major changes should include an architecture decision record in `docs/adr/`.
