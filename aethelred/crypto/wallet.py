"""Compatibility wallet exports for older top-level SDK imports."""

from aethelred.core.wallet import CompositeSignature, DualKeyWallet

# Legacy alias expected by ``aethelred.__init__`` and downstream imports.
Wallet = DualKeyWallet

__all__ = ["Wallet", "DualKeyWallet", "CompositeSignature"]
