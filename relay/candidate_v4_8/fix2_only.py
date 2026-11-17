"""v4.8b = v4.7c + fix 2 (table-label-aware absence) only."""
from relay.candidate_v4_8.verifier import build

VERSION = "v4.8b"


def verify_for(question: str):
    return build(question, fix1=False, fix2=True, version=VERSION)
