"""Bridge runner inputs to CAI's official messages, with fixed manifest examples."""
import hashlib
import json
from pathlib import Path
from . import prompts


def official_prompt_builder(sample, strategy):
    # Verify the fixed example artifact; never select examples from current inputs.
    root = Path(prompts.PROJECT_ROOT) / "prompts"
    manifest = json.loads((root / "prompt_manifest.json").read_text(encoding="utf-8"))
    artifact = root / "few_shot_examples.json"
    if (manifest["prompt_version"] != prompts.PROMPT_VERSION or
            hashlib.sha256(artifact.read_text(encoding="utf-8").encode("utf-8")).hexdigest() !=
            manifest["files_sha256"][artifact.name]):
        raise ValueError("Official prompt manifest/example integrity mismatch")
    examples = prompts.load_examples(artifact)
    if [e["sample_id"] for e in examples] != manifest["example_ids_in_order"]:
        raise ValueError("Official few-shot example order mismatch")
    return prompts.build_messages(strategy, sample.get("environment"),
                                  sample.get("instruction"), examples=examples)
