"""Foreground entrypoint for bounded LM storage-fault conformance."""

import argparse
import json
from pathlib import Path

from .transformer_lm_smoke import run_lm_smoke


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("readiness", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scp-host", default="kona")
    args = parser.parse_args(argv)
    result = run_lm_smoke(args.readiness, args.registry, args.target,
                          args.output, scp_host=args.scp_host)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
