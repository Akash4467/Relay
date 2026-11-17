"""v4.8a = v4.7c + fix 1 (check-mark condition) only."""
from relay.candidate_v4_8.verifier import build

VERSION = "v4.8a"


def verify_for(question: str):
    return build(question, fix1=True, fix2=False, version=VERSION)
