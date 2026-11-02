"""Manuscript-only contract checks. No inference, datasets or campaign execution."""
from pathlib import Path
import ast
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'publications/tecs_software_review_2026-11-02'


def test_all_manuscript_citations_have_unique_bibliography_entries():
    files = list((PAPER / 'overleaf').rglob('*.tex'))
    text = '\n'.join(p.read_text() for p in files)
    cited = {key for group in re.findall(r'\\cite\{([^}]+)\}', text)
             for key in group.split(',')}
    entries = re.findall(r'@\w+\{([^,]+),', (PAPER / 'overleaf/references.bib').read_text())
    assert len(entries) == len(set(entries)) == 12
    assert cited == set(entries)


def test_main_section_structure_and_no_internal_scientific_labels():
    sections = sorted((PAPER / 'overleaf/sections').glob('*.tex'))
    assert len(sections) == 11
    text = '\n'.join(p.read_text() for p in sections)
    assert len(re.findall(r'\\section\{', text)) == 12  # ten main + two appendices
    assert not re.search(r'\bM(?:9|10|11)(?:\.\d+)?\b|\bS[2-9]\b|pool_\d+', text)
    assert 'ideal' in text.lower() and 'ROC-AUC' in text
    assert '3,136' in text and 'partial-tier' in text


def test_exact_int8_worked_example():
    def signed(code):
        return code - 256 if code >= 128 else code
    code = (-3) & 255
    assert f'{code:08b}' == '11111101'
    assert signed(code ^ 1) / 16 == -.25
    assert signed(code ^ 128) / 16 == 7.8125
    assert (signed(code ^ 128) - signed(code)) / 16 == 8


def test_complete_tiers_do_not_depend_on_parameter_tiebreak():
    words = 16
    ordered = [(word, bit) for bit in range(7, -1, -1) for word in range(words)]
    reversed_words = [(word, bit) for bit in range(7, -1, -1) for word in reversed(range(words))]
    for count in (words, 2 * words):
        assert set(ordered[:count]) == set(reversed_words[:count])
    for count in (words // 2, words + words // 2):
        assert set(ordered[:count]) != set(reversed_words[:count])


def test_manuscript_derivation_imports_no_model_execution():
    source = ast.parse((PAPER / 'build_evidence.py').read_text())
    imported = []
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or '')
    assert not any(name.startswith(('tensorflow', 'keras', 'netsurf')) for name in imported)


def test_claim_matrix_complete_and_explicit_about_limits():
    with (PAPER / 'CLAIM_EVIDENCE.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    assert len({r['claim_id'] for r in rows}) == len(rows)
    assert all(r['source_or_receipt'] and r['scope_limit'] and None not in r for r in rows)
