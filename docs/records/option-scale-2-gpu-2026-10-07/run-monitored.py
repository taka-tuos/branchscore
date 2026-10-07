import argparse, json, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
out=Path('docs/records/option-scale-2-gpu-2026-10-07');out.mkdir(exist_ok=True)
cmd=a.command;cmd=cmd[1:] if cmd and cmd[0]=='--' else cmd
samples=[];start=time.monotonic()
with (out/(a.name+'.log')).open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 while True:
  r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
  samples.append({'elapsed_s':time.monotonic()-start,'csv':r.stdout.strip(),'returncode':r.returncode})
  if proc.poll() is not None:break
  time.sleep(.1)
result={'command':cmd,'returncode':proc.returncode,'elapsed_s':time.monotonic()-start,'sampling':'nvidia-smi every >=100ms; sampled device total, not exact allocation peak','samples':samples}
(out/(a.name+'-gpu.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'name':a.name,'returncode':proc.returncode,'elapsed_s':result['elapsed_s']}),flush=True)
raise SystemExit(proc.returncode)
