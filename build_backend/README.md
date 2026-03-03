# Vendored Python Build Backend

This directory contains a minimal vendored PEP 517 build backend used by `sdk/python/pyproject.toml`.

## Why it exists

The standard backend (`hatchling`) requires downloading build dependencies into an isolated environment. In restricted/offline environments this fails and blocks:

```bash
python -m build
```

The vendored backend removes that network dependency and allows isolated wheel/sdist builds using only repository-local code.

## Scope

- Builds pure-Python wheel (`py3-none-any`)
- Builds source distribution (`.tar.gz`)
- Reads metadata from `pyproject.toml` (PEP 621)
- Supports console scripts and optional dependencies metadata

This backend is intentionally minimal and tailored to the Aethelred Python SDK packaging workflow.
