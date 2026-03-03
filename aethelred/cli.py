"""Console entrypoint for the Aethelred Python SDK package."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from aethelred import __version__
from aethelred.core.client import AethelredClient
from aethelred.core.config import Config, Network


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aethelred", description="Aethelred Python SDK CLI utilities")
    parser.add_argument("--version", action="store_true", help="Print SDK version and exit")

    sub = parser.add_subparsers(dest="command")

    health = sub.add_parser("health", help="Check node health via the Python SDK")
    health.add_argument("--network", choices=[n.value for n in Network], default=Network.TESTNET.value)
    health.add_argument("--rpc-url", help="Override RPC URL")
    health.add_argument("--json", action="store_true", help="Output JSON")

    return parser


def _cmd_health(args: argparse.Namespace) -> int:
    network = Network(args.network)
    cfg = Config.from_network(network)
    if args.rpc_url:
        cfg.rpc_url = args.rpc_url
    client = AethelredClient(cfg)
    healthy = client.health_check()
    payload: dict[str, Any] = {
        "healthy": bool(healthy),
        "network": cfg.network.value,
        "rpc_url": cfg.rpc_url,
        "sdk_version": __version__,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"healthy={payload['healthy']} network={payload['network']} rpc={payload['rpc_url']}")
    return 0 if healthy else 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.command == "health":
        return _cmd_health(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
