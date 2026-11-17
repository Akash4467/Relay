import json, sys
sys.path.insert(0, ".")
from relay.config import app_config
from relay.verify import nli, verifier as v42
cfg = app_config(); t = cfg["verification"]["entail_min"]
m = nli.load_nli(cfg["verification"]["nli_model"])
C = {json.loads(l)["chunk_id"]: json.loads(l) for l in open("data/processed/chunks_256.jsonl", encoding="utf-8")}
def run(label, ans, ids):
    rep = v42.verify(ans, [C[i] for i in ids], set(C), m, t, t)
    print(f"{'ACCEPT' if rep.ok else 'reject'} {label:10s} " + " | ".join(f"{s.entailment if s.entailment is not None else -1:.2f}" for s in rep.sentences), "::", ans[:110])
cases = [
 ("q15 orig", "The original Transformer was trained on eight NVIDIA P100 GPUs [AIAYN:5.2:c026].", ["AIAYN:5.2:c026"]),
 ("q15 auth", "The authors trained their Transformer models on one machine with 8 NVIDIA P100 GPUs [AIAYN:5.2:c026].", ["AIAYN:5.2:c026"]),
 ("q15 auth2", "The Transformer paper's models were trained on one machine with 8 NVIDIA P100 GPUs [AIAYN:5.2:c026].", ["AIAYN:5.2:c026"]),
 ("q15 mo", "The Moshi paper states that all models are trained on H100 GPUs [MOSHI:4.4:c062].", ["MOSHI:4.4:c062"]),
 ("q15 mo2", "All Moshi models are trained on H100 GPUs [MOSHI:4.4:c062].", ["MOSHI:4.4:c062"]),
 ("q04 orig", "Mimi operates at a frame rate of 12.5 Hz, uses 8 quantizers each with a codebook size of 2048, and this configuration yields a bitrate of 1.1 kbps [MOSHI:3.3.1:c030].", ["MOSHI:3.3.1:c030"]),
 ("q04 auth", "The authors use Q = 8 quantizers, each with a codebook size of N_A = 2048 [MOSHI:3.3.1:c030]. At 12.5 Hz, this represents a bitrate of 1.1 kbps [MOSHI:3.3.1:c030].", ["MOSHI:3.3.1:c030"]),
 ("n10 orig", "The base Transformer uses 8 attention heads [AIAYN:T3:r01]. Its development BLEU score is 25.8 [AIAYN:T3:r01].", ["AIAYN:T3:r01"]),
 ("n10 row", "In Table 3, the base configuration has h = 8 [AIAYN:T3:r01]. In Table 3, the base configuration has a development BLEU of 25.8 [AIAYN:T3:r01].", ["AIAYN:T3:r01"]),
 ("n10 row2", "The base configuration in Table 3 uses h = 8 attention heads and reaches a development BLEU of 25.8 [AIAYN:T3:r01].", ["AIAYN:T3:r01"]),
 ("q13 orig", "This H-Net performance is equal to that of the Transformer configuration matched to GPT-3 XL, which also records an average accuracy of 55.5 % [HNET:T2:r11].", ["HNET:T2:r11"]),
 ("q13 both", "The 2-stage H-Net at GPT-3 Large compute and the Transformer matched to GPT-3 XL both have an average accuracy of 55.5 [HNET:T2:r10][HNET:T2:r11].", ["HNET:T2:r10", "HNET:T2:r11"]),
 ("q07 enc", "The encoder is composed of a stack of N = 6 identical layers, each with a multi-head self-attention sub-layer and a position-wise fully connected feed-forward sub-layer [AIAYN:3.1:c009].", ["AIAYN:3.1:c009"]),
 ("q07 dec", "In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack [AIAYN:3.1:c009].", ["AIAYN:3.1:c009"]),
 ("q07 qkv", "In encoder-decoder attention layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder [AIAYN:3.2.3:c016].", ["AIAYN:3.2.3:c016"]),
 # false-acceptance probes in the new phrasing
 ("FA1", "The authors trained their Transformer models on one machine with 16 NVIDIA P100 GPUs [AIAYN:5.2:c026].", ["AIAYN:5.2:c026"]),
 ("FA2", "In Table 3, the base configuration has h = 16 [AIAYN:T3:r01].", ["AIAYN:T3:r01"]),
 ("FA3", "The authors use Q = 16 quantizers, each with a codebook size of N_A = 2048 [MOSHI:3.3.1:c030].", ["MOSHI:3.3.1:c030"]),
 ("FA4", "In Table 3, the base configuration has a development BLEU of 26.4 [AIAYN:T3:r01].", ["AIAYN:T3:r01"]),
 ("FA5", "All Moshi models are trained on TPU v5 pods [MOSHI:4.4:c062].", ["MOSHI:4.4:c062"]),
]
for c in cases: run(*c)
