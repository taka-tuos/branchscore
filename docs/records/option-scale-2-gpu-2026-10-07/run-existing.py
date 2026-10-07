import subprocess,sys
from pathlib import Path
root=Path.cwd();out=root/'docs/records/option-scale-2-gpu-2026-10-07'
def run(name,cmd):
 return subprocess.call([sys.executable,str(out/'run-monitored.py'),name,'--',*map(str,cmd)])
run('ctest-e2b',['ctest','--test-dir','build-cuda','--output-on-failure','-j','1'])
for size in ['e2b','e4b']:
 model=root.parent/'models'/size/f'gemma-4-{size.upper()}-it-Q4_K_M.gguf';mmproj=model.parent/'mmproj-F16.gguf'
 if size=='e4b':
  for test in ['tokenizer','gemma4_decision_engine','prefill']:
   cmd=[root/'build-cuda'/f'branchscore_{test}_test',model]
   if test!='tokenizer':cmd += [mmproj,'cuda']
   run(f'{test}-{size}',cmd)
 for fixture in ['quality','vision','vision-text-controls']:
  run(f'{fixture}-{size}',[root/'build-cuda/branchscore-bench','--model',model,'--mmproj',mmproj,'--backend','cuda','--input',root/f'fixtures/phase4-step4-{fixture}.jsonl','--output',out/f'{fixture}-{size}.jsonl','--warmup','1'])
