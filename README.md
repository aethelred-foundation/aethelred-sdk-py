<h1 align="center">aethelred-sdk-py</h1>

<p align="center">
  <strong>Official Python SDK for the Aethelred blockchain</strong>
</p>

<p align="center">
  <a href="https://github.com/aethelred-foundation/aethelred-sdk-py/actions/workflows/repo-security-baseline.yml"><img src="https://img.shields.io/github/actions/workflow/status/aethelred-foundation/aethelred-sdk-py/repo-security-baseline.yml?branch=main&style=flat-square&label=Security" alt="Security"></a>
  <a href="https://github.com/aethelred-foundation/aethelred-sdk-py/actions/workflows/docs-hygiene.yml"><img src="https://img.shields.io/github/actions/workflow/status/aethelred-foundation/aethelred-sdk-py/docs-hygiene.yml?branch=main&style=flat-square&label=Docs+Hygiene" alt="Docs Hygiene"></a>
  <a href="https://pypi.org/project/aethelred"><img src="https://img.shields.io/pypi/v/aethelred?style=flat-square&logo=pypi" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License"></a>
</p>
<p align="center">
  <img src="https://img.shields.io/pypi/pyversions/aethelred?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PQC-Kyber+Dilithium-purple?style=flat-square" alt="PQC">
  <a href="https://docs.aethelred.io/sdk"><img src="https://img.shields.io/badge/docs-SDK-orange?style=flat-square" alt="Docs"></a>
</p>

---

## Install

```bash
pip install aethelred
```

## Quick Start

```python
from aethelred import AethelredClient, Wallet

# Connect to testnet
client = AethelredClient("https://rpc.testnet.aethelred.io")

# Load wallet
wallet = Wallet.from_mnemonic("your twelve word mnemonic...")

# Submit an AI compute job
job = client.pouw.submit_job(
    model_hash="abc123...",
    input_data=b'{"prompt": "Hello AI"}',
    verification_type="hybrid",   # "tee" | "zkml" | "hybrid"
    priority="standard",
    signer=wallet,
)
print(f"Job ID: {job.job_id}")

# Wait for seal
import time
seal = None
for _ in range(30):
    seal = client.seal.get_seal_by_job(job.job_id)
    if seal:
        break
    time.sleep(2)

print(f"Output hash: {seal.output_hash.hex()}")
print(f"Agreement: {seal.agreement_power}/{seal.total_power}")
```

## API Reference

```python
from aethelred import AethelredClient

client = AethelredClient(rpc_url)

# PoUW module
client.pouw.submit_job(model_hash, input_data, verification_type, priority, signer)
client.pouw.get_job(job_id)
client.pouw.list_jobs(submitter=None, status=None)

# Seal module
client.seal.get_seal(seal_id)
client.seal.get_seal_by_job(job_id)
client.seal.verify_seal(seal_id, output_hash)

# Bank module
client.bank.send(from_wallet, to_address, amount_uaethel)
client.bank.balance(address)
```

Full API docs: [docs.aethelred.io/sdk/python](https://docs.aethelred.io/sdk/python)

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy aethelred/
```
