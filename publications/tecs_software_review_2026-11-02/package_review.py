"""Package this manuscript and its derived receipts; never execute a model.

Refuses an existing export directory. Compile and inspect before invoking.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archive(path, files):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, source in sorted(files.items()):
            z.write(source, name)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pdf = HERE / 'overleaf/build/main.pdf'
    log = (HERE / 'overleaf/build/main.log').read_text()
    assert pdf.is_file()
    assert not any(token in log for token in ('undefined', 'Overfull', 'Fatal error'))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = {}
    for p in (HERE / 'overleaf').rglob('*'):
        rel = p.relative_to(HERE / 'overleaf')
        if p.is_file() and 'build' not in rel.parts and (
            p.suffix in ('.tex', '.bib') or p.name == '.latexmkrc' or
            (rel.parts[0] == 'figures' and p.suffix in ('.pdf', '.png'))
        ):
            source[str(rel)] = p
    source['README.md'] = HERE / 'OVERLEAF_README.md'
    archive(out / 'NetSuRF_Software_Overleaf.zip', source)
    shutil.copy2(pdf, out / 'NetSuRF_Software_Internal_Review.pdf')

    review = {str(p.relative_to(HERE)): p for p in HERE.iterdir()
              if p.is_file() and p.suffix in ('.md', '.csv', '.py')}
    for p in (HERE / 'evidence').iterdir():
        if p.is_file():
            review['evidence/' + p.name] = p
    docs = ['PAPER1_ORIGINAL_TO_CURRENT_AUDIT.md', 'PAPER1_EVIDENCE_INVENTORY.md',
            'PAPER1_NOVELTY_AUDIT.md', 'SCIENTIFIC_CONTRACTS.md',
            'MLPERF_TINY_M9_STATIC_ESTIMATORS.md', 'MLPERF_TINY_M93_MULTIBENCH.md',
            'M91_PUBLICATION_EVIDENCE.md', 'M97_FKERAS_REPRODUCTION.md',
            'LEGACY_ESTIMATOR_RECONSTRUCTION.md']
    for name in docs:
        p = ROOT / 'docs' / name
        assert p.is_file(), p
        review['docs/' + name] = p
    review['workflows/paper1_evidence_audit.py'] = ROOT / 'workflows/paper1_evidence_audit.py'
    for name in ('test_paper1_evidence_audit.py', 'test_paper1_manuscript.py'):
        review['tests/' + name] = ROOT / 'tests' / name
    for name in ('NetSuRF_Software_Overleaf.zip', 'NetSuRF_Software_Internal_Review.pdf'):
        review[name] = out / name
    manifest = {
        'kind': 'software_only_manuscript_delivery', 'version': 1,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'tracked_dirty': bool(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).strip()),
        'new_scientific_inferences': 0, 'scientific_source_writes': 0,
        'files': {name: digest(p) for name, p in sorted(review.items())},
        'overleaf_members': {name: digest(p) for name, p in sorted(source.items())},
    }
    receipt = out / 'delivery_manifest.json'
    receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    review['delivery_manifest.json'] = receipt
    archive(out / 'NetSuRF_Software_Review_Package.zip', review)
    for p in sorted(out.iterdir()):
        print(digest(p), p.name, p.stat().st_size)


if __name__ == '__main__':
    main()
