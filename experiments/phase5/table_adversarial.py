"""Phase 5 (offline): adversarial cases for experiment T on the real retrieved Table 8 rows (MOSHI:T8:r01, r02) and
HNET Table 2. T may only accept a claim whose every number is the named row's cell in the named column; anything else
must fall back to NLI (return None). Writes experiments/phase5/out/table_adversarial_{UTC}.json."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase5")
from table_check import table_supported  # noqa: E402

C = {json.loads(l)["chunk_id"]: json.loads(l) for l in open("data/processed/chunks_256.jsonl", encoding="utf-8")}
T8 = [C["MOSHI:T8:r01"], C["MOSHI:T8:r02"]]
R01, R02 = [C["MOSHI:T8:r01"]], [C["MOSHI:T8:r02"]]
H2 = [C[i] for i in ("HNET:T2:r01", "HNET:T2:r10", "HNET:T2:r15")]
CASES = [  # (description, sentence, cited, passages, should_accept)
    ("q12 redraft, correct", "Moshi with Inner Monologue achieves 26.6 % on Web Questions, 62.3 % on Llama Questions, and 22.8 % on Audio TriviaQA [MOSHI:T8:r01].", R01, T8, True),
    ("plain row, correct", "Moshi achieves 26.6 on Web Q. [MOSHI:T8:r01].", R01, T8, True),
    ("no-IM row, correct", "Moshi without Inner Monologue achieves 9.2 % on Web Questions [MOSHI:T8:r02].", R02, T8, True),
    ("wrong value", "Moshi with Inner Monologue achieves 26.6 % on Web Questions and 21.0 % on Llama Questions [MOSHI:T8:r01].", R01, T8, False),
    ("values swapped between columns", "Moshi achieves 62.3 % on Web Questions and 26.6 % on Llama Questions [MOSHI:T8:r01].", R01, T8, False),
    ("wrong variant (no-IM claim, IM numbers)", "Moshi without Inner Monologue achieves 26.6 % on Web Questions [MOSHI:T8:r01].", R01, T8, False),
    ("wrong variant (IM claim, no-IM numbers)", "Moshi with Inner Monologue achieves 9.2 % on Web Questions [MOSHI:T8:r02].", R02, T8, False),
    ("wrong entity", "SpeechGPT achieves 26.6 % on Web Questions [MOSHI:T8:r01].", R01, T8, False),
    ("comparative", "Moshi achieves 26.6 % on Web Questions, higher than SpeechGPT [MOSHI:T8:r01].", R01, T8, False),
    ("negation", "Moshi does not achieve 26.6 % on Web Questions [MOSHI:T8:r01].", R01, T8, False),
    ("extra unsupported condition", "Moshi achieves 26.6 % on Web Questions in a five-shot setting [MOSHI:T8:r01].", R01, T8, False),
    ("number without its column", "Moshi achieves 26.6 % [MOSHI:T8:r01].", R01, T8, False),
    ("missing cell claimed", "AudioLM achieves 7.0 % on Audio Trivia QA [MOSHI:T8:r01].", R01, T8, False),
    ("multi-row claim", "The 2-stage H-Net reaches 55.5 at GPT-3 Large compute, compared with 53.3 for the BPE Transformer [HNET:T2:r10][HNET:T2:r01].", H2[:2], H2, False),
    ("wrong size row", "The 2-stage H-Net matched to GPT-3 Large reaches an average accuracy of 58.2 [HNET:T2:r15].", [H2[2]], H2, False),
]
rows = []
for desc, s, cited, passages, want in CASES:
    got = table_supported(s, cited, passages)
    rows.append({"case": desc, "sentence": s, "should_accept": want, "accepted": bool(got), "reason": got, "ok": bool(got) == want})
    print(f"{'PASS' if bool(got) == want else 'FAIL'}  accept={bool(got)!s:5} want={want!s:5} {desc}" + (f"  [{got[:70]}]" if got else ""))
print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} as intended")
out = Path(f"experiments/phase5/out/table_adversarial_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with out.open("x", encoding="utf-8") as f:
    json.dump(rows, f, indent=1, ensure_ascii=False)
print(out)
