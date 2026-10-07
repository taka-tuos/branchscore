"""Measured prompt-order comparison only; no runtime renderer/cache changes.

Run from branchscore with GPU access. --prepare-only emits inputs/provenance.
Existing completed runs are retained; --report-only recomputes the summary.
"""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).resolve().parent
OUT = RECORD / 'image-placement'
TEMP = Path('/tmp/branchscore-gpu-image-placement')
IMAGE = '<|image><|image|><image|>\n'
END = '<turn|>\n<|turn>model\n'
BINARIES = {'f32': Path('/tmp/branchscore-gpu-prefill-probe'),
            'flash': Path('/tmp/branchscore-gpu-prefill-flash-probe')}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare():
    OUT.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    rows = [json.loads(s) for s in (ROOT / 'fixtures/phase4-step4-vision.jsonl').read_text().splitlines()]
    labels = [s.split('\t')[1] for s in (ROOT / 'docs/records/phase-4-plus-option-label-candidates.tsv').read_text().splitlines()[1:]]
    cases = []
    for row in rows:
        opts = [{'description': o['description'], 'letter': chr(65+i)} for i, o in enumerate(row['options'])]
        prompt = ('<bos><|turn>system\nApply the supplied criterion to the supplied evidence. '
                  'Choose exactly one listed option. Respond with only its uppercase letter, '
                  'with no explanation or reasoning.<turn|>\n<|turn>user\n' + IMAGE +
                  'State:\n' + row['state'] + '\n\nQuestion:\n' + row['question'] + '\n\nOptions:\n' +
                  json.dumps(opts, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + END)
        cases.append(dict(row, name=row['id'], image=str((ROOT/'fixtures'/row['image']).resolve()),
                          labels=[o['letter'] for o in opts], first_prompt=prompt,
                          original_renderer='gemma4-categorical-v1'))
    for row in json.loads((RECORD/'inputs.json').read_text()):
        if not row['name'].startswith('scale-image-'):
            continue
        cases.append(dict(row, id=row['name'], labels=labels[:len(row['options'])],
                          first_prompt=(RECORD/(row['name']+'.prompt')).read_text(),
                          original_renderer=row['renderer']))
    inputs = []
    for case in cases:
        first = case.pop('first_prompt')
        assert first.count(IMAGE) == 1 and first.endswith(END)
        for placement in ['first', 'last']:
            # Last-image prompts add one separating newline after the JSON array.
            prompt = first if placement == 'first' else first.replace(IMAGE, '', 1)[:-len(END)] + '\n' + IMAGE + END
            name = case['name'] + '-image-' + placement
            (OUT/(name+'.prompt')).write_text(prompt)
            (OUT/(name+'.labels')).write_text('\n'.join(case['labels'])+'\n')
            inputs.append(dict(case, name=name, placement=placement,
                               renderer=case['original_renderer'] if placement == 'first' else 'research-gemma4-image-last-v1',
                               prompt_sha256=sha(OUT/(name+'.prompt')),
                               image_sha256=sha(Path(case['image']))))
    dump(OUT/'inputs.json', inputs)
    provenance = json.loads((RECORD/'manifest.json').read_text())
    model_files = {p: meta for p, meta in provenance['models'].items() if '/e4b/' in p}
    for p, meta in model_files.items():
        assert sha(Path(p)) == meta['sha256'], 'model bytes changed: ' + p
    tracked = [Path(__file__), RECORD/'prefill-probe.cpp', RECORD/'run-monitored.py',
               ROOT/'docs/records/phase4-step4-cpu-reference/prefill-cd-flash.cpp',
               ROOT/'docs/records/phase4-step4-cpu-reference/state-cache-cd-f16.cpp',
               ROOT/'src/prefill_engine.cpp', ROOT/'src/state_cache.cpp',
               ROOT/'build-cuda/libbranchscore_core.so']
    dump(OUT/'manifest.json', {
        'project_revision': provenance['project_revision'], 'ggml_revision': provenance['ggml_revision'],
        'models': model_files, 'source_and_core_sha256': {str(p): sha(p) for p in tracked},
        'binary_sha256': {path: sha(binary) for path, binary in BINARIES.items()},
        'fixture_sha256': sha(ROOT/'fixtures/phase4-step4-vision.jsonl'),
        'comparison': 'image first versus image last; same descriptions/labels/image; last adds newline before image',
        'cache_reuse': False, 'warmup_count': 0, 'repetitions': 1, 'request_cases': len(cases),
        'planned_runs': len(inputs)*len(BINARIES),
        'timing': 'prefill_ms excludes Vision/load; monitored process time includes load and embedding download/disk writes',
    })
    print('prepared', len(inputs)*len(BINARIES), 'runs', flush=True)
    return inputs


def run(inputs):
    model = ROOT.parent/'models/e4b/gemma-4-E4B-it-Q4_K_M.gguf'
    mmproj = model.parent/'mmproj-F16.gguf'
    failures = []
    for case in inputs:
        for path, binary in BINARIES.items():
            name = case['name']+'-e4b-'+path
            result_file = OUT/(name+'.json')
            if result_file.exists():
                continue
            base = TEMP/name
            cmd = [str(binary), str(model), str(mmproj), str(OUT/(case['name']+'.prompt')),
                   str(OUT/(case['name']+'.labels')), case['image'], str(base)]
            rc = subprocess.call([sys.executable, str(RECORD/'run-monitored.py'), 'image-placement/'+name, '--', *cmd], cwd=ROOT)
            if rc:
                failures.append(name)
                continue
            meta = json.loads(Path(str(base)+'.json').read_text())
            assert all(math.isfinite(x) for x in meta['logits']+meta['probabilities'])
            meta.update(case_id=case['id'], placement=case['placement'], renderer=case['renderer'],
                        expected_selected_id=case['expected_selected_id'],
                        selected_id=case['options'][int(meta['selected_index'])]['id'],
                        options=case['options'], prompt_sha256=case['prompt_sha256'], image_sha256=case['image_sha256'])
            for extension in ['before', 'after', 'answers', 'embeddings']:
                src = Path(str(base)+'.'+extension)
                meta[extension+'_sha256'] = sha(src)
                if extension != 'embeddings':
                    (OUT/(name+'.'+extension)).write_bytes(src.read_bytes())
            dump(result_file, meta)
    if failures:
        raise RuntimeError('failed runs: ' + ', '.join(failures))


def report():
    inputs = json.loads((OUT/'inputs.json').read_text())
    results = {}
    for case in inputs:
        for path in BINARIES:
            name = case['name']+'-e4b-'+path
            result = json.loads((OUT/(name+'.json')).read_text())
            gpu = json.loads((OUT/(name+'-gpu.json')).read_text())
            assert gpu['returncode'] == 0
            used = [int(s['csv'].split(',')[3].strip()) for s in gpu['samples'] if s['returncode']==0 and s['csv']]
            result['sampled_peak_mib'] = max(used)
            assert abs(sum(result['probabilities'])-1) < 1e-6
            results[(case['id'], case['placement'], path)] = result
    # Image-first inputs must exactly reproduce the production tokenizer fixture.
    baseline = {x['request_id']: x for s in (RECORD/'vision-e4b.jsonl').read_text().splitlines()
                if (x := json.loads(s)).get('kind') == 'decision'}
    baseline_checks = []
    for case_id, old in baseline.items():
        for path in BINARIES:
            name = case_id+'-image-first-e4b-'+path
            before = [int(x) for x in (OUT/(name+'.before')).read_text().splitlines()]
            after = [int(x) for x in (OUT/(name+'.after')).read_text().splitlines()]
            # Placeholder is replaced by visual embeddings in the probe.
            assert before + [258880] + after == old['prompt']['prompt_token_ids']
            src = RECORD/('vision-e4b'+('-flash' if path=='flash' else '')+'.jsonl')
            previous = next(x for s in src.read_text().splitlines()
                            if (x := json.loads(s)).get('request_id') == case_id)
            current = results[(case_id, 'first', path)]
            delta = max(abs(a-b['raw_score']) for a,b in zip(current['logits'], previous['options']))
            baseline_checks.append(dict(case_id=case_id, path=path, selected_match=current['selected_id']==previous['selected_id'], max_raw_delta=delta))
    pairs = []
    for case_id in dict.fromkeys(x['id'] for x in inputs):
        for path in BINARIES:
            a, b = [results[(case_id, placement, path)] for placement in ['first','last']]
            assert a['embeddings_sha256'] == b['embeddings_sha256']
            ma, mb = statistics.mean(a['logits']), statistics.mean(b['logits'])
            pairs.append(dict(case_id=case_id, path=path, expected_id=a['expected_selected_id'],
                              first_id=a['selected_id'], last_id=b['selected_id'],
                              first_correct=a['selected_id']==a['expected_selected_id'],
                              last_correct=b['selected_id']==b['expected_selected_id'],
                              max_centered_delta=max(abs((x-ma)-(y-mb)) for x,y in zip(a['logits'],b['logits'])),
                              max_probability_delta=max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities'])),
                              first_positions=a['positions'], last_positions=b['positions'],
                              first_prefill_ms=a['prefill_ms'], last_prefill_ms=b['prefill_ms'],
                              first_peak_mib=a['sampled_peak_mib'], last_peak_mib=b['sampled_peak_mib']))
    summary = {}
    for path in BINARIES:
        small = [p for p in pairs if p['path']==path and p['case_id'].startswith('vision-')]
        summary[path] = dict(cases=len(small), first_correct=sum(p['first_correct'] for p in small),
                             last_correct=sum(p['last_correct'] for p in small),
                             selection_changes=sum(p['first_id']!=p['last_id'] for p in small),
                             correct_to_wrong=sum(p['first_correct'] and not p['last_correct'] for p in small),
                             wrong_to_correct=sum(not p['first_correct'] and p['last_correct'] for p in small))
    dump(OUT/'report.json', dict(successful_runs=len(results), baseline_checks=baseline_checks, summary=summary, pairs=pairs))
    print(json.dumps({'successful_runs':len(results),'summary':summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        report()
    else:
        cases = prepare()
        if not args.prepare_only:
            run(cases)
            report()
