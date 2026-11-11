"""Build an explicit upload inventory and scan it without revealing secrets.

Read-only with respect to source/experiment files; writes local audit metadata
under ignored results/upload_audit. Does not stage, commit, push or call a model.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/upload_audit'
GIT = ['git', '-c', 'safe.directory=' + ROOT.as_posix()]
R1 = 'group13-full-pilot-qwen200-20261111T090248Z-76877106'
R2 = 'group13-full-pilot-round2-20261111T153626Z-bc85f07d'
ANALYSIS = 'results/analysis/round1-vs-round2-20261111T155346Z'
REPORT = 'results/reports'

def git(*args):
    return subprocess.check_output(GIT + list(args), cwd=ROOT)

def choose_files():
    selected = {
        '.gitattributes', '.gitignore', 'src/run_experiment.py',
        'docs/pilot_round2_comparison.md', 'docs/qwen_pilot_reviewed_recovery.md',
        'docs/github_experiment_handoff.md', 'scripts/dev_acceptance.py',
        'scripts/full_pilot.py', 'scripts/pilot_round2.py', 'scripts/resume_full_pilot.py',
        'scripts/prepare_github_handoff.py', 'tests/test_dev_safety.py',
        'tests/test_full_pilot.py', 'tests/test_pilot_round2.py', 'tests/test_resume_full_pilot.py',
    }
    # Every file in the historical source seal is required for Round 2 checks.
    for run in [R1, R2]:
        selected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'results/pilot'/run).rglob('*') if p.is_file())
    selected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/ANALYSIS).iterdir() if p.is_file())
    for name in ['full_pilot_preparation_baseline_20261111.json', 'qwen_reviewed_stop_verification_20261111.json']:
        selected.add('results/analysis/' + name)
    for directory in ['results/dev/group13-dev-qwen20-20261111T060727Z-d9300025',
                      'results/analysis/group13-dev-acceptance-20261111-verified']:
        selected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).iterdir()
                        if p.is_file() and p.suffix.lower() not in {'.png', '.svg'})
    for name in ['generate_report.py', 'README.md', 'Group13_Qwen_Plus_Experimental_Results.docx',
                 'Group13_Qwen_Plus_Experimental_Results.pdf', 'figure_manifest.json', 'integrity.json',
                 'artifact_sha256.json', 'package_versions.json', 'quality_assurance.json']:
        selected.add(REPORT + '/' + name)
    selected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/REPORT/'figures').glob('[0-9][0-9]_*.svg'))
    for name in ['metrics_by_round_strategy.csv', 'classification_diagnostics.csv',
                 'prediction_distribution.csv', 'tokens_latency_stability.csv', 'agreement_by_strategy.csv',
                 'aligned_predictions.csv', 'disagreements.csv', 'error_cases.csv', 'figure_source_data.csv']:
        selected.add(REPORT + '/data/' + name)
    assert all((ROOT/p).is_file() for p in selected)
    assert not any('/_dependencies/' in p or '/_qa/' in p or '__pycache__' in p or p.endswith('.zip') for p in selected)
    return sorted(selected)

def scan_bytes(data, path, environment_secrets):
    # DOCX is an archive, so inspect every XML/rels payload rather than its
    # compressed bytes only. PDF text is inspected when a PDF reader is present.
    chunks = [data]
    if path.endswith(('.json', '.jsonl')):
        try:
            decoded=data.decode('utf-8-sig')
            records=[json.loads(line) for line in decoded.splitlines() if line.strip()] if path.endswith('.jsonl') else [json.loads(decoded)]
        except (ValueError, UnicodeError):
            return 'unreadable structured evidence'
        def has_credential(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if re.fullmatch(r'api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token',key,re.I):
                        # These sealed Pilot records use authorization for the
                        # user's English approval statement, not an HTTP header.
                        approval_note=(key=='authorization' and path.startswith('results/pilot/')
                            and Path(path).name in {'pilot_manifest.json','reviewed_failure_authorization.json','reviewed_stop_audit.json'}
                            and isinstance(item,str) and len(item.split())>=10
                            and any(word in item.lower() for word in ['user','preparation'])
                            and any(word in item.lower() for word in ['review','authorized','authorization','manually'])
                            and not item.lower().startswith(('bearer ','basic ')))
                        if not approval_note and item not in (None,'','[REDACTED]'):
                            return True
                    if has_credential(item):return True
            elif isinstance(value,list):
                return any(has_credential(item) for item in value)
            return False
        if any(has_credential(record) for record in records):
            return 'nonempty credential field in structured evidence'
    if path.endswith('.docx'):
        with zipfile.ZipFile(BytesIO(data)) as z:
            chunks += [z.read(n) for n in z.namelist() if n.endswith(('.xml','.rels'))]
    elif path.endswith('.pdf'):
        try:
            import pymupdf
            pdf = pymupdf.open(stream=data, filetype='pdf')
            chunks.append('\n'.join(p.get_text() for p in pdf).encode())
        except ImportError:
            raise RuntimeError('PDF text scanner unavailable; add local report dependencies to PYTHONPATH')
    patterns = [
        rb'\bsk-[A-Za-z0-9_-]{20,}\b',
        rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b',
        rb'\bAKIA[A-Z0-9]{16}\b',
        rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        rb'(?i)Bearer\s+[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]+){0,2}',
        rb'(?i)(?:DASHSCOPE_API_KEY|OPENAI_API_KEY)\s*[=:]\s*[\x22\x27]?sk-[A-Za-z0-9_-]+',
    ]
    for chunk in chunks:
        if any(re.search(pattern, chunk) for pattern in patterns):
            return 'credential-pattern match'
        if any(secret in chunk for secret in environment_secrets):
            return 'environment credential value match'
    return None

def main():
    sys.path.insert(0, str(ROOT/REPORT/'_dependencies'))
    OUT.mkdir(exist_ok=True)
    files = choose_files()
    env = [v.encode() for k,v in os.environ.items() if re.search(r'API.?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL', k, re.I)
           and len(v) >= 12 and k not in {'CODEX_SANDBOX_NETWORK_DISABLED'}]
    inventory=[]
    issues=[]
    for path in files:
        data=(ROOT/path).read_bytes()
        if path.endswith('/.lock') and data != b'0':
            issues.append(dict(path=path,reason='unexpected historical lock content'))
        if len(data)>10*1024*1024:
            issues.append(dict(path=path,reason='file exceeds 10 MiB review threshold'))
        if Path(path).name.startswith(('.env','~$')) or Path(path).suffix.lower() in {'.pyc','.pyd','.dll','.exe'}:
            issues.append(dict(path=path,reason='excluded file type'))
        issue=scan_bytes(data,path,env)
        if issue:issues.append(dict(path=path,reason=issue))
        inventory.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    # Do not print matched text, environment values or even substrings of them.
    (OUT/'security_scan.json').write_text(json.dumps(dict(files_scanned=len(files),issues=issues,
        status='STOP' if issues else 'PASS',docx_xml_and_pdf_text_scanned=True),indent=2)+'\n',encoding='utf-8')
    if issues:
        print(json.dumps(dict(status='STOP',suspect_files=issues),indent=2))
        raise SystemExit(2)
    (OUT/'selected_files.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
    (OUT/'selected_paths.txt').write_text(''.join(p+'\n' for p in files),encoding='utf-8')
    (OUT/'selected_pathspec.nul').write_bytes(b'\0'.join(p.encode() for p in files)+b'\0')
    with (OUT/'upload_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader();writer.writerows(inventory)
    existing=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))-{''}
    assert existing <= set(files), 'Unreviewed pre-existing staged paths; stop without unstaging them'
    print(json.dumps(dict(status='PASS',selected_files=len(files),total_bytes=sum(r['bytes'] for r in inventory),
        largest=sorted(inventory,key=lambda r:r['bytes'],reverse=True)[:8]),indent=2))

if __name__=='__main__':main()
