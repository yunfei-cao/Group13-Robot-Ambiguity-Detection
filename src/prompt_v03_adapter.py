"""Verified, source-only bridge for CAI commit 0a216fc; never edits its bundle."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {'baseline': 'v0.1-dev', 'candidate': 'v0.3-candidate'}
# SHA-256 of the actual Git blobs at CAI commit 0a216fc, not checkout bytes.
ANCHORS = {
    'baseline': {
        'prompts.py': '5e7866fb9accb9143344e21f3c8054867f46c3a29d4b549941dedbb18d352167',
        'prompts/prompt_manifest.json': '1e4db3fc47f3979dda3d2a6b7f204dcbab6516858558eb3aa8af00668762fab9'},
    'candidate': {
        'prompts.py': '6036089326376de45086954b7238da6cfc9cca6662fd83abc4b1906b5ab9a914',
        'prompts/prompt_manifest.json': 'f838f7eaf0d62b96f785e53d1ef0abc053217465186fbd35a5e0e14420e322ae'}}
ASSETS = {'zero_shot.txt', 'few_shot.txt', 'checklist.txt',
          'few_shot_examples.json', 'preview_messages.json'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


class VerifiedBundle:
    """Validate every asset before execution; native CAI guards run on LF copies.

    Only CRLF -> LF is accepted. BOMs, bare CRs and all other edits fail closed.
    Neither explicit examples nor patched hash functions bypass CAI's own loader.
    """
    def __init__(self, variant, bundle_root=None):
        if variant not in VERSIONS:
            raise ValueError('Unknown prompt variant')
        self.variant = variant
        folder = Path(bundle_root or ROOT / 'prompt_v0_3') / variant
        self.audit = {}
        normalized = {}

        def verify(relative, expected):
            raw = (folder / relative).read_bytes()
            lf = raw.replace(b'\r\n', b'\n')
            entry = dict(raw_sha256=sha(raw), lf_sha256=sha(lf),
                         expected_lf_sha256=expected, crlf_count=raw.count(b'\r\n'),
                         normalization='CRLF -> LF only')
            if entry['lf_sha256'] != expected:
                raise ValueError(f'{variant}/{relative}: normalized SHA256 mismatch')
            self.audit[relative] = entry
            normalized[relative] = lf

        for relative, expected in ANCHORS[variant].items():
            verify(relative, expected)
        self.manifest = json.loads(normalized['prompts/prompt_manifest.json'])
        if (self.manifest['prompt_version'] != VERSIONS[variant]
                or set(self.manifest['files_sha256']) != ASSETS):
            raise ValueError('Unexpected CAI manifest version/assets')
        for name, expected in self.manifest['files_sha256'].items():
            verify('prompts/' + name, expected)
        # Complete validation precedes executing even the pinned Python source.
        self._temporary = tempfile.TemporaryDirectory(prefix='group13-verified-lf-')
        self.folder = Path(self._temporary.name)
        try:
            for relative, data in normalized.items():
                target = self.folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            source = self.folder / 'prompts.py'
            module = types.ModuleType('group13_verified_' + variant)
            module.__file__ = str(source)
            exec(compile(normalized['prompts.py'], str(source), 'exec'), module.__dict__)
            self.module = module
            self.examples = module.load_examples()  # Original default-path hash guard.
            if ([e['sample_id'] for e in self.examples] != self.manifest['example_ids_in_order']
                    or [e['pair_id'] for e in self.examples] != self.manifest['example_pair_ids_in_order']
                    or [e['gold_label'] for e in self.examples] != self.manifest['example_labels_in_order']):
                raise ValueError('CAI example order/identity mismatch')
            for strategy in module.STRATEGIES:
                exported = normalized['prompts/' + strategy + '.txt'].decode('utf-8')
                if module.system_prompt(strategy) + '\n' != exported:
                    raise ValueError('Exported text differs from CAI module')
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._temporary.cleanup()

    def __call__(self, sample, strategy):
        if set(sample) != {'environment', 'instruction'}:
            raise ValueError('Only environment/instruction may enter prompt builder')
        # Do not pass examples explicitly: retain CAI's native default loader guard.
        return deepcopy(self.module.build_messages(strategy, **sample))
