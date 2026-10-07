import hashlib,json,subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'docs/records/option-scale-2-gpu-2026-10-07';temp=Path('/tmp/branchscore-gpu-probes');temp.mkdir(exist_ok=True)
image_scale='--image-scale' in sys.argv
inputs=[x for x in json.loads((out/'inputs.json').read_text()) if x['name'].startswith('scale-image-')==image_scale]
for size in (['e4b'] if image_scale else ['e4b','e2b']):
 model=root.parent/'models'/size/f'gemma-4-{size.upper()}-it-Q4_K_M.gguf';mmproj=model.parent/'mmproj-F16.gguf'
 for path,binary in [('f32','/tmp/branchscore-gpu-prefill-probe'),('flash','/tmp/branchscore-gpu-prefill-flash-probe')]:
  for case in inputs:
   name=f"{case['name']}-{size}-{path}";base=temp/name
   cmd=[binary,str(model),str(mmproj),str(out/(case['name']+'.prompt')),str(out/(case['name']+'.labels')),case['image'],str(base)]
   rc=subprocess.call([sys.executable,str(out/'run-monitored.py'),name,'--',*cmd])
   if not rc:
    meta=json.loads(Path(str(base)+'.json').read_text());meta['expected_selected_id']=case['expected_selected_id'];meta['selected_id']=case['options'][int(meta['selected_index'])]['id']
    for extension in ['before','after','answers','embeddings']:
     f=Path(str(base)+'.'+extension)
     if f.exists():meta[extension+'_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
    (out/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
   else:
    print('FAILED',name,'continue independent cases',flush=True)
