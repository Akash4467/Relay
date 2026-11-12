import pytest

from relay.retrieval.paper_mentions import named_papers


@pytest.mark.parametrize("question,papers", [
    ("How is input segmented in the original Transformer compared with H-Net?", ["AIAYN", "HNET"]),
    ("Which GPUs trained the original Transformer and Moshi?", ["AIAYN", "MOSHI"]),
    ("According to Vaswani et al., what is d_k?", ["AIAYN"]),
    ("In Attention Is All You Need, how many heads?", ["AIAYN"]),
    ("What does HNet's router do?", ["HNET"]),
    # bare "Transformer" is a component or baseline name in the other papers, not a reference to AIAYN
    ("How does H-Net compare with the BPE Transformer?", ["HNET"]),
    ("What does Moshi's RQ-Transformer do?", ["MOSHI"]),
    ("What chunk size does Moshi's dynamic chunking layer use?", ["MOSHI"]),
    ("How do attention heads work?", []),
])
def test_named_papers(question, papers):
    assert named_papers(question) == papers
