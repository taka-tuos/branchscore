"""Focused live HTTP 512 mapping/admission/recovery checks on one CUDA server."""
import base64
import hashlib
import http.client
import json
import math
import re
import socket
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'models/e4b/gemma-4-E4B-it-Q4_K_M.gguf'
rows={r['id']:r for s in (OUT/'results.jsonl.effective-inputs.jsonl').read_text().splitlines() if (r:=json.loads(s))}
core={r['id']:r for s in (OUT/'results.jsonl').read_text().splitlines() if (r:=json.loads(s))}

def payload(row):
    # Using description as the key with null value preserves the core's visible
    # text exactly. Lexical order of these identifiers matches the input order.
    body={'model':'branchscore-local','state':row['state'],'questions':{'q':{
        'type':'choice','instructions':row['question'],'criteria':{o['description']:None for o in row['options']}}}}
    if row.get('image'):
        body['image']={'media_type':'image/png','data_base64':base64.b64encode(Path(row['image']).read_bytes()).decode()}
    return body

with socket.socket() as reservation:
    reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
command=[str(ROOT/'build-cuda/branchscore-server'),'--model',str(MODEL),'--mmproj',str(MODEL.parent/'mmproj-F16.gguf'),
         '--backend','cuda','--host','127.0.0.1','--port',str(port)]
checks=[]
def call(name,body=None,path='/v1/systemone',expected_status=200,expected_error=None):
    start=time.monotonic();conn=http.client.HTTPConnection('127.0.0.1',port,timeout=90)
    conn.request('GET' if body is None else 'POST',path,body=None if body is None else json.dumps(body),
                 headers={} if body is None else {'Content-Type':'application/json'})
    response=conn.getresponse();raw=response.read();conn.close()
    assert response.status==expected_status,(name,response.status,raw[:300])
    data=json.loads(raw)
    if expected_error:assert data['error']['code']==expected_error,(name,data)
    checks.append(dict(name=name,status=response.status,elapsed_s=time.monotonic()-start,
                       response_sha256=hashlib.sha256(raw).hexdigest(),error=expected_error))
    (OUT/('http-'+name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    return data

with (OUT/'http-server.log').open('w') as log:
    proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    try:
        for attempt in range(100):
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.2):pass
                break
            except OSError:
                if proc.poll() is not None:raise RuntimeError('server exited during startup')
                time.sleep(.1)
        health=call('health',path='/healthz')
        assert health['limits']=={'max_options':512,'max_prefill_positions':16384,'max_request_prefill_positions':32768}
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5);conn.request('GET','/ui')
        response=conn.getresponse();html=response.read().decode();conn.close()
        assert response.status==200 and 'const maxOptions = 512;' in html and 'id="bulk-options"' in html
        assert '__BRANCHSCORE_MAX_OPTIONS__' not in html
        javascript=re.search(r'<script>(.*?)</script>',html,re.S)[1]
        (OUT/'served-ui.js').write_text(javascript)
        subprocess.run(['node','--check',str(OUT/'served-ui.js')],check=True)
        for name in ['text-17','text-512','mpn-digits-fhd']:
            body=payload(rows[name]);response=call(name,body)
            options=rows[name]['options'];probabilities=response['answers']['q']['probabilities']
            assert len(probabilities)==len(options) and all(math.isfinite(p) for p in probabilities.values())
            assert abs(sum(probabilities.values())-1)<1e-6
            expected=next(o['description'] for o in options if o['id']==core[name]['selected_id'])
            assert response['answers']['q']['choice']==expected
            diagnostic=response['branchscore']['questions']['q']
            assert diagnostic['renderer_id']=='gemma4-categorical-512-v1'
            assert diagnostic['option_order']==[o['description'] for o in options]
            assert response['branchscore']['prefill_positions']==core[name]['positions']
            assert diagnostic['timings_ms']['prefill_attention_path']=='flash'
            assert max(abs(diagnostic['raw_logits'][o['description']]-s['raw_score']) for o,s in zip(options,core[name]['scores']))<1e-4
        too_many=payload(rows['text-512']);too_many['questions']['q']['criteria']['EXTRA CANDIDATE']=None
        call('reject-513',too_many,expected_status=422,expected_error='invalid_request')
        call('reject-positions',payload(rows['reject-16385-fhd']),expected_status=413,expected_error='token_budget_exceeded')
        aggregate=payload(rows['mpn-digits-fhd']);question=aggregate['questions']['q']
        aggregate['questions']={str(i):question for i in range(3)}
        call('reject-aggregate',aggregate,expected_status=413,expected_error='request_token_budget_exceeded')
        recovery=call('recovery',payload(rows['recovery-small']))
        assert recovery['answers']['q']['choice']=='Keep it running'
        assert recovery['branchscore']['questions']['q']['renderer_id']=='gemma4-categorical-v1'
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
(OUT/'http-checks.json').write_text(json.dumps(dict(command=command,checks=checks,
    ui_javascript_syntax=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(checks,indent=2))
