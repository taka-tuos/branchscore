import json,subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'docs/records/option-scale-2-gpu-2026-10-07';temp=Path('/tmp/branchscore-gpu-probes')
for size in ['e4b','e2b']:
 model=root.parent/'models'/size/f'gemma-4-{size.upper()}-it-Q4_K_M.gguf';mmproj=model.parent/'mmproj-F16.gguf'
 for path in ['f32','flash']:
  for case in json.loads((out/'inputs.json').read_text()):
   name=f"{case['name']}-{size}-{path}";base=temp/name
   if not (out/(name+'.json')).exists():continue
   completed=out/('reference-'+name+'-gpu.json')
   if completed.exists() and json.loads(completed.read_text())['returncode']==0:continue
   image=case['image']!='-';binary='/tmp/branchscore-reference-'+('image' if image else 'text')+('-flash' if path=='flash' else '')
   if image:cmd=[binary,str(model),str(mmproj),str(base)+'.before',str(base)+'.after',str(base)+'.answers',str(base)+'.embeddings','0']
   else:cmd=[binary,str(model),str(base)+'.before',str(base)+'.reference-logits','0','0']
   subprocess.call([sys.executable,str(out/'run-monitored.py'),'reference-'+name,'--',*cmd])
 for fixture in ['quality','vision']:
  subprocess.call([sys.executable,str(out/'run-monitored.py'),f'{fixture}-{size}-flash','--','/tmp/branchscore-bench-flash','--model',str(model),'--mmproj',str(mmproj),'--backend','cuda','--input',str(root/f'fixtures/phase4-step4-{fixture}.jsonl'),'--output',str(out/f'{fixture}-{size}-flash.jsonl'),'--warmup','1'])
