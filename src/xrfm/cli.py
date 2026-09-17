"""Minimal XRFM command-line interface.

Phase 0 scope — deliberately tiny (boring and explicit):

    xrfm info                      # package version + packaged default config
    xrfm validate-config PATH      # load + validate a YAML config, print a summary

Training/generation entry points remain the thin scripts under ``scripts/``
(they need dataset/tokenizer arguments that belong to a repo checkout, not
to an installed library). Do not grow this CLI into product surface in
Phase 0; if it grows later, keep it delegating to package APIs only.
"""

from __future__ import annotations

import argparse
import json
import sys

import xrfm
from xrfm.config import default_config, load_config


def _cmd_info(_args: argparse.Namespace) -> int:
    cfg = default_config()
    print(f"xrfm {xrfm.__version__}")
    print(f"python {sys.version.split()[0]}")
    print(f"default preset : {cfg.project}")
    print(
        "default model  : d_model={d_model} n_layers={n_layers} n_heads={n_heads} "
        "d_ff={d_ff} max_seq_len={max_seq_len} vocab={vocab_size}".format(**vars(cfg.model))
    )
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # ConfigError / YAML errors → invalid
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xrfm", description="XR Foundation Model CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show package version and packaged default config")

    vc = sub.add_parser("validate-config", help="validate a YAML config file")
    vc.add_argument("path", help="path to the YAML config")

    args = parser.parse_args(argv)
    if args.command == "info":
        return _cmd_info(args)
    if args.command == "validate-config":
        return _cmd_validate_config(args)
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
