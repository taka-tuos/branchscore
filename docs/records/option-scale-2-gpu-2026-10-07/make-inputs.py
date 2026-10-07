"""Deterministic research inputs; no change to the production A-P renderer."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
labels = [x.split('\t')[1] for x in (root/'docs/records/phase-4-plus-option-label-candidates.tsv').read_text().splitlines()[1:]]
manifest=[]
def write(name,row,answer_labels,renderer):
    opts=[{'letter':label,'description':option['description']} for label,option in zip(answer_labels,row['options'])]
    text='<bos><|turn>system\nApply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase '
    text+=('letter' if renderer=='gemma4-categorical-v1' else 'label')+', with no explanation or reasoning.<turn|>\n<|turn>user\n'
    image=(root/'fixtures'/row['image']).resolve() if row.get('image') else None
    if image:text+='<|image><|image|><image|>\n'
    text+='State:\n'+row['state']+'\n\nQuestion:\n'+row['question']+'\n\nOptions:\n'
    text+=json.dumps(opts,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'<turn|>\n<|turn>model\n'
    (out/(name+'.prompt')).write_text(text)
    (out/(name+'.labels')).write_text('\n'.join(answer_labels)+'\n')
    manifest.append({'name':name,'renderer':renderer,'image':str(image) if image else '-', 'options':row['options'],'expected_selected_id':row['expected_selected_id']})
quality=list(map(json.loads,(root/'fixtures/phase4-step4-quality.jsonl').read_text().splitlines()))
for row in quality:
    if row['id'] in ['long-route-16','priority-ja','close-scores']:
        write('control-'+row['id'],row,list('ABCDEFGHIJKLMNOP')[:len(row['options'])],'gemma4-categorical-v1')
vision=list(map(json.loads,(root/'fixtures/phase4-step4-vision.jsonl').read_text().splitlines()))
for row in vision:
    if row['id'] in ['vision-board-base','vision-status-base','vision-table-base']:
        write('control-'+row['id'],row,list('ABCDEFGHIJKLMNOP')[:len(row['options'])],'gemma4-categorical-v1')
for n in [16,384,512]:
    row={'state':'Exactly one item has assigned color blue. All other items have assigned color red.',
         'question':'Which listed item has assigned color blue?',
         'options':[{'id':f'item{i:03d}','description':f'Item {i:03d}: the assigned color is '+('blue.' if i==7 else 'red.')} for i in range(n)],
         'expected_selected_id':'item007'}
    for reverse in [False,True]:
        case=dict(row,options=list(reversed(row['options'])) if reverse else row['options'])
        write(f'scale-{n}'+('-reversed' if reverse else ''),case,labels[:n],'research-gemma4-two-letter-v1')
board = next(x for x in vision if x['id']=='vision-board-base')
for n in [384,512]:
    options=board['options']+[{'id':f'zone{i:03d}','description':f'The selected region is ZONE {i:03d}.'} for i in range(n-len(board['options']))]
    for reverse in [False,True]:
        case=dict(board,options=list(reversed(options)) if reverse else options)
        write(f'scale-image-{n}'+('-reversed' if reverse else ''),case,labels[:n],'research-gemma4-two-letter-v1')
(out/'inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('prepared',len(manifest),'research inputs')
