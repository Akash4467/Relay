from relay.config import app_config
from relay.llm import prompts


def p(cid, paper, sid, title, text):
    return {"chunk_id": cid, "paper": paper, "section_id": sid, "section_title": title, "text": text}


PASSAGES = [p("MOSHI:3.3.1:c030", "MOSHI", "3.3.1", "Architecture", "At 12.5Hz, this represents 1.1kbps."),
            p("MOSHI:T4:r01", "MOSHI", "5.2", "Audio Tokenization", "Mimi: bitrate = 1.1kbps")]


def test_prompt_version_matches_config():
    assert app_config()["prompt_version"] == prompts.PROMPT_VERSION


def test_messages_follow_design_format():
    msgs = prompts.messages("What bitrate?", PASSAGES)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "reply with exactly: NOT_IN_PAPERS" in msgs[0]["content"]
    assert msgs[1]["content"] == (
        "Question: What bitrate?\n\nPassages:\n"
        "[MOSHI:3.3.1:c030] (MOSHI §3.3.1 Architecture)\nAt 12.5Hz, this represents 1.1kbps.\n\n"
        "[MOSHI:T4:r01] (MOSHI §5.2 Audio Tokenization)\nMimi: bitrate = 1.1kbps")


def test_redraft_appends_draft_and_failures():
    msgs = prompts.redraft_messages("q", PASSAGES, "It is 2kbps [MOSHI:T4:r01].",
                                    [("It is 2kbps [MOSHI:T4:r01].", "number 2 not in cited passage")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert '- "It is 2kbps [MOSHI:T4:r01]." — number 2 not in cited passage' in msgs[-1]["content"]


def test_refusal_names_what_was_searched_without_duplicates():
    text = prompts.refusal_text(PASSAGES + PASSAGES[:1])
    assert text == ("The three papers don't state this. I searched: MOSHI §3.3.1 Architecture; "
                    "MOSHI §5.2 Audio Tokenization.")
    assert prompts.refusal_text([]) == "The three papers don't state this."


def test_busy_text():
    assert "about 12 seconds" in prompts.busy_text(12)
