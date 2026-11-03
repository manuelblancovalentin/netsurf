"""Verify and archive an immutable v3 derived database without editing it."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from netsurf.manifests import sha256_file
from netsurf.reliability_database_v3 import SCHEMA_VERSION


def export(product: Path, destination: Path) -> None:
    product = product.resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    checks = json.loads((product/"checksums.json").read_text())
    if checks["schema_version"] != SCHEMA_VERSION:
        raise ValueError("not a v3 reliability product")
    if (product/"FINALIZED").read_text().strip() != f"reliability_database_sha256={checks['files']['reliability_v3.sqlite']}":
        raise ValueError("derived product finalization marker differs")
    for relative, digest in checks["files"].items():
        if sha256_file(product/relative) != digest:
            raise ValueError(f"derived product artifact changed: {relative}")
    archives = {}
    for flavor in ("compact", "complete"):
        path = destination/f"{product.name}_{flavor}.zip"
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(product.rglob("*")):
                if not file.is_file():
                    continue
                relative = file.relative_to(product)
                if flavor == "compact" and relative.name == "reliability_v3.sqlite":
                    continue
                archive.write(file, arcname=str(Path(product.name)/relative))
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"archive failed integrity test: {path}")
        archives[flavor] = {"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size}
    (destination/"export_receipt.json").write_text(json.dumps({
        "schema_version":"1.0.0", "product":str(product),
        "product_database_sha256":checks["files"]["reliability_v3.sqlite"], "archives":archives,
    }, indent=2, sort_keys=True)+"\n")
    (destination/"mac_scp_commands.txt").write_text(
        "# Run on the Mac with the researcher's existing 'kona' SSH alias; replace it if needed.\n"
        + "\n".join(f"scp kona:{record['path']} ~/Downloads/" for record in archives.values())+"\n"
    )


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    export(args.product,args.destination)
    print(args.destination/"export_receipt.json")


if __name__ == "__main__":
    main()
