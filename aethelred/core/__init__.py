"""Core module for Aethelred SDK.

Provides the primary wallet, client, and configuration types.

Example::

    from aethelred.core import DualKeyWallet, AethelredConfig

    config = AethelredConfig(endpoint="https://mainnet.aethelred.org")
    wallet = DualKeyWallet()
    print(wallet.address)
"""

from aethelred.core.config import AethelredConfig, SecretStr
from aethelred.core.exceptions import (
    AethelredError,
    ProofError,
    ValidationError,
)

try:
    from aethelred.core.wallet import DualKeyWallet, CompositeSignature, ECDSASigner
except Exception:  # Optional crypto backends may be absent in minimal installs
    DualKeyWallet = None  # type: ignore[assignment]
    CompositeSignature = None  # type: ignore[assignment]
    ECDSASigner = None  # type: ignore[assignment]

__all__ = [
    "AethelredConfig",
    "SecretStr",
    "AethelredError",
    "ProofError",
    "ValidationError",
]

if DualKeyWallet is not None:
    __all__.extend(["DualKeyWallet", "CompositeSignature", "ECDSASigner"])
