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
            prefix = 'evidence/superseded_readiness/' if p.name == 'transformer_readiness_update.json' else 'evidence/'
            review[prefix + p.name] = p
    docs = ['PAPER1_ORIGINAL_TO_CURRENT_AUDIT.md', 'PAPER1_EVIDENCE_INVENTORY.md',
            'PAPER1_NOVELTY_AUDIT.md', 'SCIENTIFIC_CONTRACTS.md',
            'MLPERF_TINY_M9_STATIC_ESTIMATORS.md', 'MLPERF_TINY_M93_MULTIBENCH.md',
            'M91_PUBLICATION_EVIDENCE.md', 'M97_FKERAS_REPRODUCTION.md',
            'LEGACY_ESTIMATOR_RECONSTRUCTION.md', 'PAPER1_EXPANDED_CAMPAIGN.md']
    for name in docs:
        p = ROOT / 'docs' / name
        assert p.is_file(), p
        review['docs/' + name] = p
    review['workflows/paper1_evidence_audit.py'] = ROOT / 'workflows/paper1_evidence_audit.py'
    for name in ('paper1_campaign.py','run_paper1_resnet8.sh','run_paper1_kws.sh','run_paper1_toycar.sh'):
        review['workflows/'+name]=ROOT/'workflows'/name
    review['scripts/preflight_paper1_kws_v2.py']=ROOT/'scripts/preflight_paper1_kws_v2.py'
    for name in ('test_paper1_evidence_audit.py', 'test_paper1_manuscript.py',
                 'test_paper1_campaign.py','test_streaming_protection_gate.py',
                 'test_paper1_distilbert_evidence.py'):
        review['tests/' + name] = ROOT / 'tests' / name
    review['netsurf/streaming_validation.py']=ROOT/'netsurf/streaming_validation.py'
    for name in ('paper1_campaigns.json','paper1_fkeras_source.json','paper1_distilbert_source.json','paper1_target_status.json',
                 'benchmark_expansion_registry.json'):
        review['config/'+name]=ROOT/'config'/name
    for name in ('resnet8','kws','toycar'):
        name=f'paper1_{name}_common_grid_v1.toml'
        review['config/experiments/'+name]=ROOT/'config/experiments'/name
    review['config/experiments/paper1_kws_common_grid_v2.toml']=ROOT/'config/experiments/paper1_kws_common_grid_v2.toml'
    for name in ('canonical-tests.log','canonical-tests-with-artifacts.log','pinned-tests.log'):
        review['validation/'+name]=ROOT/'workspace'/name
    review['validation/manuscript-revision-tests.log']=ROOT/'workspace/manuscript-revision-tests.log'
    review['validation/manuscript-revision-latex.log']=ROOT/'workspace/manuscript-revision-latex.log'
    distilbert=json.loads((ROOT/'config/paper1_distilbert_source.json').read_text())
    for name,key in [('complete.zip','complete_sha256'),('compact.zip','compact_sha256'),
                     ('checksums.json','checksums_sha256')]:
        path=Path(distilbert['path'])/name
        assert digest(path)==distilbert[key], f'archived DistilBERT source changed: {name}'
        review['distilbert-source/'+name]=path
    for p in (ROOT/'workspace/streaming-derived-v2').iterdir():
        if p.is_file():review['streaming-engineering/analysis/'+p.name]=p
    for name in ('paper1_streaming_gate_v1','paper1_streaming_gate_repeat_v1'):
        for p in (ROOT/'workspace/runs'/name).iterdir():
            if p.is_file():review['streaming-engineering/'+name+'/'+p.name]=p
    for p in (ROOT/'workspace/paper1-campaigns').rglob('*'):
        if p.is_file() and p.name in ('plan.json','preflight.json','STAGE_COMPLETE.json'):
            review['campaign-preflights/'+str(p.relative_to(ROOT/'workspace/paper1-campaigns'))]=p
    for name in ('NetSuRF_Software_Overleaf.zip', 'NetSuRF_Software_Internal_Review.pdf'):
        review[name] = out / name
    manifest = {
        'kind': 'software_only_manuscript_and_campaign_delivery', 'version': 3,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'tracked_dirty': bool(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).strip()),
        'manuscript_builder_inferences': 0, 'historical_scientific_source_writes': 0,
        'production_campaigns_launched': 0,
        'benchmark_expansion_gate': 'INCOMPLETE',
        'main_fkeras_definition': 'published projection; absolute component within bit tiers',
        'distilbert_scope': '1024 FP32 bits in first32 classifier.weight words, all NEGATIVE-logit row0; not whole-model protection',
        'distilbert_scientific_payload_sha256':distilbert['analysis_payload_sha256'],
        'bounded_engineering_evidence': {
            'streaming_independent_fields': 2, 'identical_execution_count': 2,
            'payload_sha256':'c8ee6163da82e8cd4fee6f13e1e9ed9eb49da6daac1dd6dc490fe49618792674'},
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
