# M31R

Offline-first Rust-focused small language model training and inference platform.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
m31r --help
m31r info
m31r crawl --config configs/global.yaml
```

## Tests

```bash
make test
```

## License

MIT — Eshan Roy
