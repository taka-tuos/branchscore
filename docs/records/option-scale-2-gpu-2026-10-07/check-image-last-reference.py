"""Post-observation diagnostic subset; not an independent quality benchmark."""
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = Path(__file__).resolve().parent
OUT = RECORD/'image-placement'
TEMP = Path('/tmp/branchscore-gpu-image-placement')
selected_cases = ['vision-board-changed', 'vision-status-base',
                  'vision-table-relayout-reversed', 'scale-image-512', 'scale-image-512-reversed']
model = ROOT.parent/'models/e4b/gemma-4-E4B-it-Q4_K_M.gguf'
comparisons = []
binaries = {}
for case in selected_cases:
    for path in ['f32','flash']:
        name = case+'-image-last-e4b-'+path
        base = TEMP/name
        binary = Path('/tmp/branchscore-reference-image'+('-flash' if path=='flash' else ''))
        binaries[str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
        monitor = OUT/('reference-'+name+'-gpu.json')
        if not monitor.exists() or json.loads(monitor.read_text())['returncode']:
            cmd = [str(binary), str(model), str(model.parent/'mmproj-F16.gguf'),
                   str(base)+'.before', str(base)+'.after', str(base)+'.answers', str(base)+'.embeddings', '0']
            subprocess.run([sys.executable, str(RECORD/'run-monitored.py'), 'image-placement/reference-'+name,
                            '--', *cmd], cwd=ROOT, check=True)
        log = (OUT/('reference-'+name+'.log')).read_text()
        assert not re.search(r'CPU_Mapped.*model buffer', log), 'unexpected CPU model placement'
        assert re.search(r'CUDA0\s+model buffer', log), 'CUDA model placement missing'
        log = re.sub(r'~llama_context:[^\n]*\n','',log)
        found = re.findall(r'answer_(\d+)\s+id=(\d+)\s+logit=([-+\deE.]+)',log)
        answers = [int(x) for x in (OUT/(name+'.answers')).read_text().split()]
        assert [int(x[0]) for x in found]==list(range(len(answers)))
        assert [int(x[1]) for x in found]==answers
        reference = [float(x[2]) for x in found]
        assert all(math.isfinite(x) for x in reference)
        branch = json.loads((OUT/(name+'.json')).read_text())
        assert hashlib.sha256(Path(str(base)+'.embeddings').read_bytes()).hexdigest()==branch['embeddings_sha256']
        idx = max(range(len(reference)), key=reference.__getitem__)
        mr,mb = statistics.mean(reference),statistics.mean(branch['logits'])
        e = [math.exp(x-max(reference)) for x in reference]
        probabilities = [x/sum(e) for x in e]
        comparisons.append(dict(name=name, reference_logits=reference, branch_logits=branch['logits'],
                                reference_selected_id=branch['options'][idx]['id'],
                                branch_selected_id=branch['selected_id'], expected_id=branch['expected_selected_id'],
                                max_centered_delta=max(abs(x-mr-y+mb) for x,y in zip(reference,branch['logits'])),
                                max_probability_delta=max(abs(x-y) for x,y in zip(probabilities,branch['probabilities']))))
output = dict(selected_after_observation=True, cases=selected_cases, comparisons=comparisons,
              reference_revision='19e28a27702117d8f2eb16b825b9a308111f67d9', binary_sha256=binaries,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              selection_matches=sum(c['reference_selected_id']==c['branch_selected_id'] for c in comparisons))
(OUT/'reference-diagnostic.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'selection_matches':output['selection_matches'],'total':len(comparisons)}))
for c in comparisons:
    print(c['name'], c['branch_selected_id'], c['reference_selected_id'], 'expected', c['expected_id'])
