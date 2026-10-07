"""Token counts only; synthetic 20-character identifiers, not real BOM accuracy."""
import hashlib,json,random,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
labels=[x.split('\t')[1] for x in (root/'docs/records/phase-4-plus-option-label-candidates.tsv').read_text().splitlines()[1:]]
rng=random.Random(20261007)
patterns={'digits':[f'{x:020d}' for x in range(512)],'alphanumeric':[''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(20)) for _ in range(512)],'separated':[''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-') for _ in range(20)) for _ in range(512)]}
results=[]
for pattern,items in patterns.items():
 for count in [256,512]:
  options=[{'description':item,'letter':label} for item,label in zip(items[:count],labels)]
  prompt='<bos><|turn>system\nApply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase label, with no explanation or reasoning.<turn|>\n<|turn>user\n<|image><|image|><image|>\nState:\nThe image shows a received component package.\n\nQuestion:\nWhich BOM entry matches the manufacturer part number printed on the package, including its suffix?\n\nOptions:\n'+json.dumps(options,sort_keys=True,separators=(',',':'))+'<turn|>\n<|turn>model\n'
  r=subprocess.run([str(root/'build-cuda/branchscore-tokenize'),'--model',str(root.parent/'models/e4b/gemma-4-E4B-it-Q4_K_M.gguf'),'--text',prompt,'--no-bos'],capture_output=True,text=True,check=True)
  tokens=int(re.search(r'token_count=(\d+)',r.stdout)[1]);results.append({'pattern':pattern,'option_count':count,'description_chars':20,'rendered_tokens_including_one_image_placeholder':tokens,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
(out/'mpn-token-counts.json').write_text(json.dumps({'purpose':'synthetic tokenizer sizing only; actual image expansion/VRAM/quality not measured','model_sha256':'85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87','seed':20261007,'results':results},indent=2)+'\n')
for x in results:print(x['pattern'],x['option_count'],x['rendered_tokens_including_one_image_placeholder'])
