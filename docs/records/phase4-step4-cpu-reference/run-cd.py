from pathlib import Path
import json,subprocess,os,time
r=Path('/home/ubuntu/branchscore'); out=Path('/tmp/step4-cd-20261002');out.mkdir(exist_ok=True)
fs=[json.loads(x) for x in (r/'fixtures/phase4-step4-vision.jsonl').read_text().splitlines()]
env=dict(os.environ,GGML_CPU_TILED_MM='0',GGML_NO_IQ_PANEL='1',OMP_NUM_THREADS='4')
model='/home/ubuntu/llama.cpp/models/gemma-4-E4B-it-Q4_K_M.gguf';mm='/home/ubuntu/llama.cpp/models/e4b-mmproj-F16.gguf'
for mode in ['C','D']:
 for f in fs:
  c=f['id'];args=[model,mm,'/tmp/'+c+'.boundary.before.ids','/tmp/'+c+'.boundary.after.ids','/tmp/'+c+'.answers.ids','/tmp/step4-shared-'+Path(f['image']).stem+'.f32']
  args+=['0'] if mode=='C' else [str(r/'fixtures'/f['image']),mode]
  start=time.monotonic()
  with (out/(mode+'-'+c+'.out')).open('w') as stdout,(out/(mode+'-'+c+'.log')).open('w') as stderr:
   p=subprocess.run(['/tmp/step4-image-cd-flash' if mode=='C' else '/tmp/step4-branch-cd-flash']+args,env=env,stdout=stdout,stderr=stderr)
  print(mode,c,'return',p.returncode,'seconds',round(time.monotonic()-start,2),flush=True)
  if p.returncode:raise SystemExit(p.returncode)
