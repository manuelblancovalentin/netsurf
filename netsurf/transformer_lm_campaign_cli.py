"""Concise foreground CLI for the targeted 96-bit LM campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifests import sha256_file
from .transformer_lm_campaign import (aggregate, export, freeze, run_final,
                                      run_selection, validate_plan, _prepare)


def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze","validate","selection","final","aggregate","export"))
    parser.add_argument("target")
    parser.add_argument("plan", type=Path)
    parser.add_argument("readiness", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--scp-host", default="kona")
    args=parser.parse_args(argv)
    if args.phase=="freeze":
        plan=freeze(args.readiness,args.registry,args.target,args.plan,
            args.plan.parent.parent.parent/"workspace"/"plans"/args.plan.stem/"NETSURF_ASSET_READINESS.json")
        result={"status":"FROZEN","plan_sha256":plan["plan_sha256"],"costs":plan["costs"]}
    elif args.phase=="validate":
        plan,_,_,_,_=_prepare(args.plan,args.readiness,args.registry)
        if plan["target"]!=args.target:raise ValueError("target differs from frozen plan")
        result={"status":"FROZEN_VALID","plan_sha256":plan["plan_sha256"],"costs":plan["costs"]}
    elif args.phase=="selection":
        result=run_selection(args.plan,args.readiness,args.registry,args.run_root)
    elif args.phase=="final":
        result=run_final(args.plan,args.readiness,args.registry,args.run_root)
    elif args.phase=="aggregate":
        report=aggregate(args.plan,args.run_root)
        result={"status":"ANALYSIS_VERIFIED","plan_sha256":report["plan_sha256"],
                "analysis":str(args.run_root/"analysis.json"),
                "analysis_sha256":sha256_file(args.run_root/"analysis.json"),
                "rows":len(report["rows"]),"comparisons":len(report["comparisons"])}
    else:
        result={"status":"EXPORTED","output":str(export(args.plan,args.readiness,
            args.registry,args.run_root,scp_host=args.scp_host))}
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
