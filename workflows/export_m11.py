"""Export M11 source and NEW evidence only; never dereference external assets.

Run with canonical Python from the implementation worktree. No inference.
The archive is a handoff, not a redistribution of datasets/checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def git(*words):
        return subprocess.check_output(["git", "-C", str(root), *words])

    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("Commit the reviewed source before exporting a release.")
    commit = git("rev-parse", "HEAD").decode().strip()
    source = []
    allowed = {"netsurf", "tests", "docs", "config", "dependencies", "workflows"}
    top = {"AGENTS.md", "README.md", "pyproject.toml", "setup.cfg", ".gitignore"}
    for name in git("ls-files", "-z").decode().split("\0"):
        if name and (Path(name).parts[0] in allowed or name in top):
            path = root / name
            if path.is_symlink():
                continue
            source.append((path, "source/" + name))

    from netsurf.scaling_study import verify_directory
    from netsurf.manifests import digest_object, sha256_file

    run_names = ["m11_historical_reaudit", "m11_oracle_analysis_v2",
                 "m11_oracle_analysis_v2-figures", "m11_kws_oracle_plan",
                 "m11_kws_oracle_selection", "m11_kws_final_replay",
                 "m11_kws_sampling_cost_plan"]
    for model in ("kws", "resnet8", "toycar"):
        run_names += [f"m11_{model}_validated_{kind}" for kind in
                      ("plan", "selection", "registry", "final")]
        run_names += [f"m11_{model}_analysis", f"m11_{model}_analysis-figures"]
        run_names += [f"m11_frozen_{model}_{axis}" for axis in
                      ("ber", "expected_applied_flips", "enriched")]
    evidence = []
    bundles = []
    for name in run_names:
        directory = root / "workspace/runs" / name
        if not directory.is_dir():
            raise SystemExit(f"Missing required delivery evidence: {directory}")
        markers = sorted(directory.rglob("FINALIZED")) + sorted(directory.rglob("FROZEN"))
        if not markers and not name.endswith("-figures"):
            raise SystemExit(f"No finalized evidence in {directory}")
        for marker in markers:
            payload = verify_directory(marker.parent, frozen=marker.name == "FROZEN")
            bundles.append({"path": str(marker.parent.relative_to(root)),
                            "scientific_payload_sha256": digest_object(payload)})
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise SystemExit(f"Unexpected evidence symlink: {path}")
            if path.is_file():
                evidence.append((path, str(path.relative_to(root))))
    files = source + evidence
    inventory = {name: sha256_file(path) for path, name in files}
    index = {"schema_version": "1.0.0", "source_commit": commit,
             "branch": git("branch", "--show-current").decode().strip(),
             "base": "10cb2f24d273cdba9c7a99e32ae207b278238f51",
             "bundles": bundles, "files": inventory,
             "limitations": ["External datasets, checkpoints and historical S5 bundles are prerequisites, not redistributed.",
                             "Absolute source paths remain pinned; copied plans do not silently relocate historical evidence.",
                             "Engineering results do not establish protection superiority or an untouched confirmation study."]}
    patch = git("format-patch", "--stdout", index["base"] + "..HEAD")
    index["patch_sha256"] = hashlib.sha256(patch).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("INDEX.json", json.dumps(index, indent=2) + "\n")
        archive.writestr("implementation.patch", patch)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity failure")
    print(json.dumps({"archive": str(output), "sha256": sha256_file(output),
                      "bytes": output.stat().st_size, "source_commit": commit,
                      "verified_bundles": len(bundles)}, indent=2))


if __name__ == "__main__":
    main()
