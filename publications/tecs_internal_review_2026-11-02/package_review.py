"""Package the completed manuscript without changing any scientific archive."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

BASE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

validation=json.loads((BASE/'evidence/FINAL_VALIDATION.json').read_text())
assert validation['undefined_citations_or_references']==0
pdf=BASE/'NetSuRF_TECS_internal_review.pdf'
overleaf_zip=BASE/'NetSuRF_TECS_overleaf.zip'
package=BASE/'NetSuRF_TECS_review_package.zip'
for p in [pdf,overleaf_zip,package,BASE/'PACKAGE_MANIFEST.json',BASE/'SHA256SUMS.txt']:
    if p.exists():
        raise FileExistsError(f'Refusing to overwrite delivered artifact: {p}')
with (BASE/'build/main.pdf').open('rb') as src,pdf.open('xb') as dst:
    shutil.copyfileobj(src,dst)
source_files=sorted(p for p in (BASE/'overleaf').rglob('*') if p.is_file())
with zipfile.ZipFile(overleaf_zip,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in source_files:
        z.write(p,p.relative_to(BASE/'overleaf'))
with zipfile.ZipFile(overleaf_zip) as z:
    assert 'main.tex' in z.namelist() and z.testzip() is None

paths=[pdf,overleaf_zip,BASE/'original_overleaf.zip',
       BASE/'README.md',BASE/'ADVERSARIAL_REVIEW.md',BASE/'BIBLIOGRAPHY_VERIFICATION.md',
       BASE/'prepare_evidence.py',BASE/'complete_review.py',BASE/'package_review.py']
paths+=sorted(p for p in (BASE/'evidence').rglob('*') if p.is_file())
manifest={
 'schema_version':'1.0.0','purpose':'complete ACM TECS internal review draft, not journal submission',
 'title':'Beyond Isolated Bit Sensitivity: Conditional Fault Vulnerability and Protection-Allocation Limits in Quantized Neural Networks',
 'scientific_source_changes':0,'new_model_inferences':0,
 'files':{str(p.relative_to(BASE)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths},
 'overleaf_files':{str(p.relative_to(BASE/'overleaf')):sha(p) for p in source_files}}
with (BASE/'PACKAGE_MANIFEST.json').open('x') as f:
    json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
paths.append(BASE/'PACKAGE_MANIFEST.json')
with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths: z.write(p,p.relative_to(BASE))
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    for rel,meta in manifest['files'].items():
        assert hashlib.sha256(z.read(rel)).hexdigest()==meta['sha256'],rel
with (BASE/'SHA256SUMS.txt').open('x') as f:
    for p in [pdf,overleaf_zip,package,BASE/'original_overleaf.zip',BASE/'PACKAGE_MANIFEST.json']:
        f.write(f'{sha(p)}  {p.name}\n')
print(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [pdf,overleaf_zip,package]},indent=2))
