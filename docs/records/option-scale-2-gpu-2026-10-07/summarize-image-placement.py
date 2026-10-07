"""Extra descriptive comparisons for the completed image-placement GPU study."""
import hashlib
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent/'image-placement'
inputs = json.loads((OUT/'inputs.json').read_text())
report = json.loads((OUT/'report.json').read_text())
results = {}
embedding_hashes = {}
for case in inputs:
    for path in ['f32','flash']:
        row = json.loads((OUT/(case['name']+'-e4b-'+path+'.json')).read_text())
        results[(case['id'],case['placement'],path)] = row
        embedding_hashes.setdefault(case['image_sha256'],set()).add(row['embeddings_sha256'])
assert all(len(hashes)==1 for hashes in embedding_hashes.values())

family_accuracy = {}
stability = {}
for path in ['f32','flash']:
    for placement in ['first','last']:
        key = path+'-'+placement
        family_accuracy[key] = {}
        for family in ['board','status','table']:
            rs = [r for (cid,p,attention),r in results.items()
                  if p==placement and attention==path and cid.startswith('vision-'+family+'-')]
            assert len(rs)==9
            family_accuracy[key][family] = sum(r['selected_id']==r['expected_selected_id'] for r in rs)
        option_invariant = evidence_changes = layout_invariant = 0
        for family in ['board','status','table']:
            for visual_variant in ['base','changed','relayout']:
                selected = [results[('vision-'+family+'-'+visual_variant+suffix,placement,path)]['selected_id']
                            for suffix in ['', '-reversed', '-distractors']]
                option_invariant += len(set(selected))==1
            for suffix in ['', '-reversed', '-distractors']:
                a,b,c = [results[('vision-'+family+'-'+variant+suffix,placement,path)]['selected_id']
                         for variant in ['base','changed','relayout']]
                evidence_changes += a!=b
                layout_invariant += a==c
        stability[key] = dict(option_variants_invariant=option_invariant,
                              changed_evidence_selection_changes=evidence_changes,
                              relayout_selection_invariant=layout_invariant, denominator=9)

fa_comparisons = {}
for placement in ['first','last']:
    pairs = []
    for case_id in dict.fromkeys(x['id'] for x in inputs):
        a,b = [results[(case_id,placement,path)] for path in ['f32','flash']]
        ma,mb = statistics.mean(a['logits']),statistics.mean(b['logits'])
        pairs.append(dict(case_id=case_id, nonfa_id=a['selected_id'], fa_id=b['selected_id'],
                          nonfa_correct=a['selected_id']==a['expected_selected_id'],
                          fa_correct=b['selected_id']==b['expected_selected_id'],
                          max_centered_delta=max(abs(x-ma-y+mb) for x,y in zip(a['logits'],b['logits'])),
                          max_probability_delta=max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))))
    small = [p for p in pairs if p['case_id'].startswith('vision-')]
    fa_comparisons[placement] = dict(
        pairs=pairs, small_selection_changes=sum(p['nonfa_id']!=p['fa_id'] for p in small),
        small_correct_to_wrong=sum(p['nonfa_correct'] and not p['fa_correct'] for p in small),
        small_wrong_to_correct=sum(not p['nonfa_correct'] and p['fa_correct'] for p in small),
        small_median_max_centered_delta=statistics.median(p['max_centered_delta'] for p in small),
        small_max_centered_delta=max(p['max_centered_delta'] for p in small),
        small_median_max_probability_delta=statistics.median(p['max_probability_delta'] for p in small),
        small_max_probability_delta=max(p['max_probability_delta'] for p in small))

summary = dict(family_accuracy=family_accuracy, stability=stability,
               fa_comparisons=fa_comparisons, embedding_image_count=len(embedding_hashes),
               all_same_image_embeddings_match=True,
               baseline_selected_matches=sum(p['selected_match'] for p in report['baseline_checks']),
               baseline_comparisons=len(report['baseline_checks']),
               baseline_max_raw_delta=max(p['max_raw_delta'] for p in report['baseline_checks']))
summary['placement_deltas'] = {}
for path in ['f32','flash']:
    pairs = [p for p in report['pairs'] if p['path']==path and p['case_id'].startswith('vision-')]
    summary['placement_deltas'][path] = dict(
        median_max_centered_delta=statistics.median(p['max_centered_delta'] for p in pairs),
        max_centered_delta=max(p['max_centered_delta'] for p in pairs),
        median_max_probability_delta=statistics.median(p['max_probability_delta'] for p in pairs),
        max_probability_delta=max(p['max_probability_delta'] for p in pairs))
summary['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='fa_comparisons'},indent=2))
