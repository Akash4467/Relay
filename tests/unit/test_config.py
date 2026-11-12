import pydantic
import pytest

from relay import config


def test_repo_config_loads():
    assert config.app_config()["corpus_version"] == "c1"
    settings = config.retrieval_settings()
    assert set(settings) == {"A", "B", "C"}  # D returns with expand.py
    assert (settings["A"].rerank, settings["B"].rerank) == (False, True)  # A vs B isolates reranking
    assert config.piece_sizes() == [256, 512]


def test_invalid_setting_rejected(tmp_path):
    (tmp_path / "retrieval_settings.yaml").write_text("X: {piece_tokens: big, candidates: 20}")
    with pytest.raises(pydantic.ValidationError):
        config.retrieval_settings(tmp_path)
