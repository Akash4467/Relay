from relay.llm import tokenizer


def test_uses_gpt_oss_encoding():
    assert tokenizer.encoding().name == "o200k_harmony"


def test_count_and_special_tokens_are_plain_text():
    assert tokenizer.count("") == 0
    assert tokenizer.count("hello world") == 2
    assert tokenizer.count("<|endoftext|>") > 1  # never treated as a special token


def test_split_by_tokens_round_trips():
    text = "word " * 50
    pieces = tokenizer.split_by_tokens(text, 7)
    assert all(tokenizer.count(p) <= 7 for p in pieces)
    assert "".join(pieces) == text
