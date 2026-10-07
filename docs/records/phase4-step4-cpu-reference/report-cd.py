from pathlib import Path
import json,struct,math,statistics,hashlib,re
ROOT=Path('/home/ubuntu/branchscore'); raw=Path('/tmp/step4-cd-20261002')
read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
def soft(a):
 e=[math.exp(x-max(a)) for x in a];return [x/sum(e) for x in e]
def delta(a,b):
 d=[y-x for x,y in zip(a,b)];off=statistics.mean(d)
 return dict(max_abs_raw=max(map(abs,d)),mean_offset=off,max_abs_centered=max(abs(x-off) for x in d),max_abs_probability=max(abs(x-y) for x,y in zip(soft(a),soft(b))))
def result(a,ids,truth):
 s=sorted(a,reverse=True);w=ids[a.index(max(a))]
 return dict(logits=a,probabilities=soft(a),selected_id=w,correct=w==truth,top_two_margin=s[0]-s[1])
base=read(ROOT/'docs/records/phase-4-plus-option-scale-2-vision-shared-reference-e4b-cpu-2026-10-01.jsonl')
m=json.loads(Path('/tmp/step4-cd-manifest.json').read_text());m['source_sha256']['report-cd.py']=sha(Path(__file__))
rows=[m]
for c in base[1:]:
 if c.get("kind") != "shared_embedding_reference_case":continue
 cid=c['request_id'];paths={mode:raw/(mode+'-'+cid+'.out') for mode in ['C','D']}
 if not all(p.exists() and len(re.findall(r'logit=([^\s]+)',p.read_text()))==len(c['answer_token_ids']) for p in paths.values()):continue
 ids=[x['id'] for x in c['results']['q4']['scores']];truth=c['expected_selected_id']
 for mode,p in paths.items():
  assert [int(x) for x in re.findall(r' id=(\d+) logit=',p.read_text())]==c['answer_token_ids']
  assert all(math.isfinite(float(x)) for x in re.findall(r'logit=([^\s]+)',p.read_text()))
 embedding_path=Path('/tmp/step4-shared-'+Path(c['image']).stem+'.f32')
 assert sha(embedding_path)==c['embedding_sha256']
 assert embedding_path.stat().st_size==300*2560*4
 values={mode:[f32(float(x)) for x in re.findall(r'logit=([^\s]+)',p.read_text())] for mode,p in paths.items()}
 values.update(A=[f32(x['logit']) for x in c['results']['bf16']['scores']],B=[f32(x['logit']) for x in c['results']['q4']['scores']])
 results={k:result(v,ids,truth) for k,v in values.items()}
 for k in ['bf16','q4']:
  assert results['A' if k=='bf16' else 'B']['selected_id']==c['results'][k]['selected_id']
  assert results['A' if k=='bf16' else 'B']['correct']==c['results'][k]['correct']
 diffs={x+'_'+y:dict(delta(values[x],values[y]),winner_changed=results[x]['selected_id']!=results[y]['selected_id'],correct_to_wrong=results[x]['correct'] and not results[y]['correct'],wrong_to_correct=not results[x]['correct'] and results[y]['correct']) for x,y in [('A','B'),('B','C'),('C','D'),('B','D')]}
 ab=diffs['A_B'];lim=0.01 if ab['max_abs_centered']<0.04 else 0.25*ab['max_abs_centered'];plim=max(0.01,0.25*ab['max_abs_probability'])
 passes={k:d['max_abs_centered']<=lim and d['max_abs_probability']<=plim and not d['correct_to_wrong'] for k,d in diffs.items() if k in ['B_C','B_D']}
 d=diffs['C_D'];passes['C_D']=d['max_abs_centered']<=0.01 and d['max_abs_probability']<=0.001 and not d['winner_changed']
 input_hashes={name:sha(Path('/tmp/'+cid+suffix)) for name,suffix in [('before_ids','.boundary.before.ids'),('after_ids','.boundary.after.ids'),('answers','.answers.ids')]}
 before=[int(x) for x in Path('/tmp/'+cid+'.boundary.before.ids').read_text().split()]
 after=[int(x) for x in Path('/tmp/'+cid+'.boundary.after.ids').read_text().split()]
 assert before+after==[x for x in c['prompt_token_ids'] if x!=258880]
 header=paths['D'].read_text().splitlines()[0]
 resources={k:float(v) if k=='prefill_ms' else int(v) for k,v in re.findall(r'(positions|cache_bytes|graph_bytes|chunks|prefill_ms)=([0-9.]+)',header)}
 assert resources['positions']==len(before)+300+len(after)
 ref_log=(raw/('C-'+cid+'.log')).read_text()
 configuration_lines=[l for l in ref_log.splitlines() if l.startswith('llama_context:') or (('buffer size' in l or 'KV self size' in l) and ('load_tensors:' in l or 'llama_kv_cache:' in l))]
 row=dict(reference_configuration_lines=configuration_lines,input_ids_sha256=input_hashes,branchscore_resources=resources,kind='cd_cpu_case',request_id=cid,expected_selected_id=truth,option_ids=ids,answer_token_ids=c['answer_token_ids'],prompt_identity=c['prompt_identity'],prompt_token_ids=c['prompt_token_ids'],embedding_sha256=c['embedding_sha256'],results=results,deltas=diffs,numeric_limits=dict(centered=lim,probability=plim),screening_pass=passes,execution_headers={k:p.read_text().splitlines()[0] for k,p in paths.items()},raw_log_sha256={k:sha(raw/(k+'-'+cid+'.log')) for k in paths},raw_output_sha256={k:sha(p) for k,p in paths.items()})
 rows.append(row)
summary=dict(kind='cd_cpu_summary',denominator=len(rows)-1,target_cases=27,correct={k:sum(x['results'][k]['correct'] for x in rows[1:]) for k in ['A','B','C','D']},comparisons={k:{metric:sum(x['deltas'][k][metric] for x in rows[1:]) for metric in ['winner_changed','correct_to_wrong','wrong_to_correct']} for k in ['A_B','B_C','C_D','B_D']},screening_pass={k:sum(x['screening_pass'][k] for x in rows[1:]) for k in ['B_C','C_D','B_D']})
summary['delta_ranges']={k:{metric:[min(x['deltas'][k][metric] for x in rows[1:]),max(x['deltas'][k][metric] for x in rows[1:])] for metric in ['max_abs_raw','max_abs_centered','max_abs_probability']} for k in ['A_B','B_C','C_D','B_D']} if len(rows)>1 else {}
rows.append(summary)
(ROOT/'docs/records/phase-4-plus-option-scale-2-cd-e4b-q4-cpu-2026-10-02.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
print(json.dumps(summary,indent=2))
for x in rows[1:-1]:print(x['request_id'],{k:r['selected_id'] for k,r in x['results'].items()},x['screening_pass'])
