"""Build temporary probes against the already-built project and pinned reference."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[3];source=Path(__file__).resolve().parent
project_lib=root/'build-cuda';ggml_lib=project_lib/'third_party/ggml/src'
common=['g++','-std=c++17','-O2','-I',str(root/'include'),'-I',str(root/'third_party/ggml/include')]
link=['-L',str(project_lib),'-lbranchscore_core','-L',str(ggml_lib),'-lggml','-lggml-base',f'-Wl,-rpath,{project_lib}:{ggml_lib}']
subprocess.run(common+[str(source/'prefill-probe.cpp'),*link,'-o','/tmp/branchscore-gpu-prefill-probe'],check=True)
objects=[]
for src,name in [('prefill-cd-flash.cpp','prefill-flash'),('state-cache-cd-f16.cpp','cache-f16')]:
 obj=f'/tmp/branchscore-{name}.o';objects.append(obj)
 subprocess.run(common+['-fPIC','-c',str(root/'docs/records/phase4-step4-cpu-reference'/src),'-o',obj],check=True)
subprocess.run(common+['-DPROBE_FLASH',str(source/'prefill-probe.cpp'),*objects,*link,'-o','/tmp/branchscore-gpu-prefill-flash-probe'],check=True)
bench_obj=project_lib/'CMakeFiles/branchscore-bench.dir/tools/branchscore_bench.cpp.o'
subprocess.run(['g++',*objects,str(bench_obj),*link,'-o','/tmp/branchscore-bench-flash'],check=True)
for kind in ['text','image','text-flash','image-flash']:
 subprocess.run(['g++','-std=c++17','-O2','-I','/tmp/branchscore-llama-reference-19e28a2/include','-I','/tmp/branchscore-llama-reference-19e28a2/ggml/include',str(source/f'reference-{kind}.cpp'),'-L','/tmp/branchscore-llama-reference-build/bin','-lllama','-lggml','-lggml-base','-Wl,-rpath,/tmp/branchscore-llama-reference-build/bin','-o',f'/tmp/branchscore-reference-{kind}'],check=True)
