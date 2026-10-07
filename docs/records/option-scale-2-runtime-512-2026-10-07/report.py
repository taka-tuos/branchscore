"""Recompute focused adoption checks from saved production measurements."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = OUT.parent / 'option-scale-2-gpu-2026-10-07'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


summary = {'fixture_regression': {}, 'scale_regression': {}}
for group in ['quality', 'vision']:
    current = [r for r in rows(OUT / (group + '-e4b.jsonl')) if r['kind'] == 'decision']
    before = {r['request_id']: r for r in rows(PRIOR / (group + '-e4b-flash.jsonl')) if r['kind'] == 'decision'}
    fixtures = {r['id']: r for r in rows(ROOT / ('fixtures/phase4-step4-' + group + '.jsonl'))}
    differences = []
    for r in current:
        b = before[r['request_id']]
        assert [o['id'] for o in r['options']] == [o['id'] for o in b['options']]
        differences.extend(abs(o['raw_score'] - old['raw_score']) for o, old in zip(r['options'], b['options']))
        assert r['selected_id'] == b['selected_id']
    summary['fixture_regression'][group] = {
        'count': len(current), 'max_candidate_raw_delta': max(differences),
        'correct': sum(r['selected_id'] == fixtures[r['request_id']]['expected_selected_id'] for r in current),
    }

probe = rows(OUT / 'results.jsonl')
assert len(probe) == 17
summary['probe'] = []
for r in probe:
    if 'scores' in r:
        assert len(r['scores']) == r['option_count']
        assert len({o['token_id'] for o in r['scores']}) == r['option_count']
        assert all(math.isfinite(o['raw_score']) and math.isfinite(o['probability']) for o in r['scores'])
        assert abs(sum(o['probability'] for o in r['scores']) - 1) < 1e-6
        assert max(r['scores'], key=lambda o: o['raw_score'])['id'] == r['selected_id']
    else:
        assert r['id'] == 'reject-16385-fhd' and r['error'] == 'token_budget_exceeded'
    summary['probe'].append({k: v for k, v in r.items() if k != 'scores'})

for r in probe:
    name = r['id']
    if name.startswith('text-'):
        source = 'scale-' + name[5:]
    elif name.startswith('scale-image-'):
        source = name
    else:
        continue
    prior_path = PRIOR / (source + '-e4b-flash.json')
    if not prior_path.exists():
        continue
    before = json.loads(prior_path.read_text())
    assert [o['label'] for o in r['scores']] == before['labels']
    delta = max(abs(o['raw_score'] - old) for o, old in zip(r['scores'], before['logits']))
    summary['scale_regression'][name] = {'max_candidate_raw_delta': delta, 'selected_matches': r['selected_id'] == before['selected_id']}

manifest = json.loads((OUT / 'manifest.json').read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(OUT / 'tokenizer-measured.cpp') == manifest['source_sha256']['src/tokenizer.cpp']
for name, digest in manifest['assets'].items():
    assert sha(OUT / name) == digest
labels = [r['label'] for r in csv.DictReader((ROOT / 'docs/records/phase-4-plus-option-label-candidates.tsv').open(), delimiter='\t')]
assert re.findall(r'"([A-Z]{2})"', (ROOT / 'src/gemma4_answer_labels.hpp').read_text()) == labels[:512]
summary['fixed_labels_match_tsv'] = True
summary['measured_tokenizer_and_assets_match_hashes'] = True
samples = json.loads((OUT / 'runtime-probe-gpu.json').read_text())
assert samples['returncode'] == 0
summary['max_device_vram_sample_mib'] = max(int(s['csv'].split(',')[3]) for s in samples['samples'] if s['returncode'] == 0)
after_large = probe[next(i for i, r in enumerate(probe) if r['id'] == 'text-512'):]
assert len({r['after_request_free_bytes'] for r in after_large}) == 1
summary['post_large_request_free_bytes'] = after_large[0]['after_request_free_bytes']
summary['http'] = json.loads((OUT / 'http-checks.json').read_text())
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ['probe', 'http']}, indent=2))
