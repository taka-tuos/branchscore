"""Recompute descriptive GPU checks; these are not optimization adoption criteria."""
import array, gzip, json, math, re, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent;temp=Path('/tmp/branchscore-gpu-probes')
def load(p):return json.loads(p.read_text())
def delta(a,b):
 d=[y-x for x,y in zip(a,b)];m=statistics.mean(d)
 def softmax(v):
  e=[math.exp(x-max(v)) for x in v];s=sum(e);return [x/s for x in e]
 pa,pb=softmax(a),softmax(b)
 return {'max_raw':max(map(abs,d)),'mean_offset':m,'max_centered':max(abs(x-m) for x in d),'max_probability':max(abs(y-x) for x,y in zip(pa,pb))}
def margin(v):return sorted(v,reverse=True)[0]-sorted(v,reverse=True)[1]
def peak(name):
 p=out/(name+'-gpu.json')
 if not p.exists():return None
 return max(int(x['csv'].split(',')[3]) for x in load(p)['samples'] if x['returncode']==0)
result={'benchmarks':{},'probes':[],'reference_comparisons':[],'execution_failures':[]}
for p in out.glob('*-gpu.json'):
 d=load(p)
 if d['returncode']:result['execution_failures'].append({'name':p.stem,'returncode':d['returncode']})
for size in ['e2b','e4b']:
 for group in ['quality','vision','vision-text-controls']:
  fixture={x['id']:x for x in map(json.loads,(root/f'fixtures/phase4-step4-{group}.jsonl').read_text().splitlines())}
  by_path={}
  for path,suffix in [('f32',''),('flash','-flash')]:
   name=f'{group}-{size}{suffix}';file=out/(name+'.jsonl')
   if not file.exists():continue
   rows=[x for x in map(json.loads,file.read_text().splitlines()) if x.get('kind')=='decision'];by_path[path]={x['request_id']:x for x in rows}
   assert set(by_path[path])==set(fixture)
   for r in rows:
    assert r['schema_version']==2 and r['scoring_basis']=='answer_slot_logit' and not r['terminator_scored']
    opts=r['options'];scores=[x['raw_score'] for x in opts];probs=[x['relative_probability'] for x in opts]
    assert all(math.isfinite(x) for x in scores+probs) and abs(sum(probs)-1)<1e-9
    idx=max(range(len(scores)),key=scores.__getitem__)
    assert r['selected_index']==idx and opts[idx]['id']==r['selected_id']
   result['benchmarks'][name]={'correct':sum(x['selected_id']==fixture[x['request_id']]['expected_selected_id'] for x in rows),'total':len(rows),'sampled_peak_mib':peak(name),'median_request_ms':statistics.median(x['measured_request_ms'] for x in rows),'wrong':[{'id':x['request_id'],'selected':x['selected_id'],'expected':fixture[x['request_id']]['expected_selected_id']} for x in rows if x['selected_id']!=fixture[x['request_id']]['expected_selected_id']],'by_variant':{}}
   for variant in sorted({x.get('variant','control') for x in fixture.values()}):
    subset=[x for x in rows if fixture[x['request_id']].get('variant','control')==variant]
    result['benchmarks'][name]['by_variant'][variant]={'correct':sum(x['selected_id']==fixture[x['request_id']]['expected_selected_id'] for x in subset),'total':len(subset)}
  if 'flash' in by_path:
   comparison=[]
   for k,a in by_path['f32'].items():
    b=by_path['flash'][k];assert a['prompt']['prompt_token_ids']==b['prompt']['prompt_token_ids']
    comparison.append(dict(id=k,selection_changed=a['selected_id']!=b['selected_id'],before_correct=a['selected_id']==fixture[k]['expected_selected_id'],after_correct=b['selected_id']==fixture[k]['expected_selected_id'],**delta([x['raw_score'] for x in a['options']],[x['raw_score'] for x in b['options']])))
   result['benchmarks'][f'{group}-{size}-optimization-deltas']=comparison
for case in load(out/'inputs.json'):
 for size in ['e2b','e4b']:
  paths={}
  for path in ['f32','flash']:
   name=f"{case['name']}-{size}-{path}";p=out/(name+'.json')
   if not p.exists():continue
   d=load(p);paths[path]=d
   row={'name':name,**d,'sampled_peak_mib':peak(name),'top_two_margin':margin(d['logits'])};result['probes'].append(row)
   ref_log=out/('reference-'+name+'.log');ref_gpu=out/('reference-'+name+'-gpu.json')
   if not ref_gpu.exists() or load(ref_gpu)['returncode']:continue
   base=temp/name
   if case['image']=='-':
    values=array.array('f');values.frombytes(Path(str(base)+'.reference-logits').read_bytes())
    answers=[int(x) for x in Path(str(base)+'.answers').read_text().split()];ref=[values[x] for x in answers]
   else:
    text=(ref_log.read_text() if ref_log.exists() else gzip.open(str(ref_log)+'.gz','rt').read())
    # Buffered stdout can be split by the two stderr destructor messages.
    text=re.sub(r'~llama_context:[^\n]*\n','',text)
    found=re.findall(r'answer_(\d+)\s+id=(\d+)\s+logit=([-+\deE.]+)',text)
    assert [int(x[0]) for x in found]==list(range(len(d['logits'])))
    assert [int(x[1]) for x in found]==[int(x) for x in Path(str(base)+'.answers').read_text().split()]
    ref=[float(x[2]) for x in found]
   assert len(ref)==len(d['logits']) and all(math.isfinite(x) for x in ref)
   idx=max(range(len(ref)),key=ref.__getitem__)
   result['reference_comparisons'].append({'name':name,'branch_logits':d['logits'],'reference_logits':ref,'branch_selected_id':d['selected_id'],'reference_selected_id':case['options'][idx]['id'],'selection_changed':idx!=d['selected_index'],'reference_sampled_peak_mib':peak('reference-'+name),**delta(ref,d['logits'])})
  if len(paths)==2:
   assert paths['f32']['before_sha256']==paths['flash']['before_sha256'] and paths['f32']['after_sha256']==paths['flash']['after_sha256']
   if case['image']!='-':assert paths['f32']['embeddings_sha256']==paths['flash']['embeddings_sha256']
result['reference_path_deltas']=[]
reference_by_name={x['name']:x for x in result['reference_comparisons']}
for case in load(out/'inputs.json'):
 for size in ['e2b','e4b']:
  key=f"{case['name']}-{size}"
  b=reference_by_name.get(key+'-f32');c=reference_by_name.get(key+'-flash')
  if not b or not c:continue
  def compare(scores_a,scores_b,id_a,id_b):
   return dict(selected_before=id_a,selected_after=id_b,selection_changed=id_a!=id_b,
     before_correct=id_a==case['expected_selected_id'],after_correct=id_b==case['expected_selected_id'],**delta(scores_a,scores_b))
  result['reference_path_deltas'].append({'name':key,
   'B_to_C':compare(b['reference_logits'],c['reference_logits'],b['reference_selected_id'],c['reference_selected_id']),
   'C_to_D':compare(c['reference_logits'],c['branch_logits'],c['reference_selected_id'],c['branch_selected_id']),
   'B_to_D':compare(b['reference_logits'],c['branch_logits'],b['reference_selected_id'],c['branch_selected_id'])})
result['summary']={'probe_count':len(result['probes']),'reference_pairs':len(result['reference_comparisons']),'reference_selection_changes':sum(x['selection_changed'] for x in result['reference_comparisons'])}
(out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result['summary']))
for name,d in result['benchmarks'].items():
 if isinstance(d,dict):print(name,f"{d['correct']}/{d['total']}",d['sampled_peak_mib'],'MiB')
