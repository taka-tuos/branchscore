"""Sequential production CUDA verification; run from branchscore with GPU access."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'models/e4b/gemma-4-E4B-it-Q4_K_M.gguf'
MMPROJ=MODEL.parent/'mmproj-F16.gguf'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()

def monitored(name,command):
    start=time.monotonic();samples=[]
    with (OUT/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while True:
            r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            samples.append({'elapsed_s':time.monotonic()-start,'csv':r.stdout.strip(),'returncode':r.returncode})
            if proc.poll() is not None:break
            time.sleep(.1)
    result=dict(command=command,returncode=proc.returncode,elapsed_s=time.monotonic()-start,
                sampling='device-wide nvidia-smi >=100ms; sampled maximum, not exact peak',samples=samples)
    (OUT/(name+'-gpu.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'name':name,'returncode':proc.returncode,'elapsed_s':result['elapsed_s']}),flush=True)
    if proc.returncode:raise RuntimeError('failed '+name)

if __name__=='__main__':
    source_files=[*ROOT.glob('src/*'),*ROOT.glob('include/branchscore/*'),*ROOT.glob('tools/*.cpp'),
                  *ROOT.glob('tests/*.cpp'),OUT/'runtime-probe.cpp',Path(__file__)]
    manifest=dict(base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  ggml_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party/ggml',text=True).strip(),
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files if p.is_file()},
                  models={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [MODEL,MMPROJ]},
                  assets={p.name:sha(p) for p in OUT.glob('*.png')},input_sha256=sha(OUT/'inputs.jsonl'),
                  build='Release, CUDA architecture 75; one model/backend/request; F16 KV/mask Flash, full SWA, 512 microbatch',
                  limits={'options':512,'prefill_positions':16384,'http_request_positions':32768},
                  binaries={str(p):sha(p) for p in [ROOT/'build-cuda/libbranchscore_core.so',ROOT/'build-cuda/branchscore-bench',ROOT/'build-cuda/branchscore-server',Path('/tmp/branchscore-runtime-512-probe')]})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,executable,args in [
        ('ctest-e2b','ctest',['--test-dir',str(ROOT/'build-cuda'),'--output-on-failure']),
        ('tokenizer-e4b',str(ROOT/'build-cuda/branchscore_tokenizer_test'),[str(MODEL)]),
        ('engine-e4b',str(ROOT/'build-cuda/branchscore_gemma4_decision_engine_test'),[str(MODEL),str(MMPROJ),'cuda']),
        ('prefill-e4b',str(ROOT/'build-cuda/branchscore_prefill_test'),[str(MODEL),str(MMPROJ),'cuda']),
    ]:
        # The full E2B CTest was already run; preserve its rerun only when this script is used independently.
        if name=='ctest-e2b' and (OUT/'ctest-e2b.log').exists():continue
        monitored(name,[executable,*args])
    for group in ['quality','vision']:
        dest=OUT/(group+'-e4b.jsonl')
        if dest.exists():continue
        monitored(group+'-e4b',[str(ROOT/'build-cuda/branchscore-bench'),'--model',str(MODEL),'--mmproj',str(MMPROJ),
                  '--backend','cuda','--input',str(ROOT/f'fixtures/phase4-step4-{group}.jsonl'),'--output',str(dest),'--warmup','1'])
    monitored('runtime-probe',['/tmp/branchscore-runtime-512-probe',str(MODEL),str(MMPROJ),str(OUT/'inputs.jsonl'),str(OUT/'results.jsonl')])
