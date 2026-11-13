"""Structured table checks (policy v3). They only reject; acceptance always needs NLI entailment."""
from relay.verify import tables


def row(config, **values):
    return {"configuration": config, "values": [
        {"metric": m.replace("__", " / ").replace("_", " "), "split": None, "column": m, "value": v}
        for m, v in values.items()]}


T2 = {"chunk_id": "AIAYN:T2:r01", "paper": "AIAYN", "kind": "table",
      "text": "Table 2: BLEU scores.", "rows": [row("Deep-Att + PosUnk", BLEU__EN_FR="39.2"),
                                                 row("GNMT + RL", BLEU__EN_DE="24.6", BLEU__EN_FR="39.92")]}
T3 = {"chunk_id": "AIAYN:T3:r01", "paper": "AIAYN", "kind": "table", "text": "Table 3: Variations.",
      "rows": [row("base", h="8", BLEU="25.8"), row("(A)", h="1", BLEU="24.9"), row("(A)", h="16", BLEU="25.8")]}
BIG = {"chunk_id": "AIAYN:T2:r05", "paper": "AIAYN", "kind": "table", "text": "Table 2: BLEU scores.",
       "rows": [{"configuration": "Transformer (big)", "values": [
           {"metric": "BLEU / EN-FR", "split": None, "column": "BLEU / EN-FR", "value": "41.8"},
           {"metric": "Training Cost (FLOPs) / EN-FR", "split": None, "column": "", "value": "2.3·10^19"}]}]}


def test_a_value_from_an_unnamed_sibling_row_is_rejected():
    assert tables.value_owner_mismatch("In Table 2, GNMT + RL reaches 39.2 BLEU on English-to-French.", [T2]) == \
        ["the value 39.2 belongs to Deep-Att + PosUnk in the cited table, not to GNMT + RL"]
    assert tables.value_owner_mismatch("In Table 2, GNMT + RL reaches 39.92 BLEU on English-to-French.", [T2]) == []


def test_unnamed_configurations_and_table_numbers_never_reject():
    # q11: rows "(A)" have no usable name, so ownership cannot be judged; NLI decides
    assert tables.value_owner_mismatch("Row (A) with h = 1 got a dev BLEU of 24.9 in Table 3.", [T3]) == []
    assert tables.value_owner_mismatch("The base configuration gets 25.8 BLEU in Table 3.", [T3]) == []
    assert tables.numbers("In Table 3, row 2 reaches 25.8") == {"25.8"}


def test_relevant_rows_are_the_named_or_quoted_ones():
    got = [r["configuration"] for r in tables.relevant_rows("h = 1 gives 24.9, the base gives 25.8.", [T3])]
    assert got == ["base", "(A)", "(A)"]  # base by name; (A) h=1 by 24.9; (A) h=16 also reports 25.8
    assert [r["configuration"] for r in tables.relevant_rows("GNMT + RL scores well.", [T2])] == ["GNMT + RL"]


def test_q11_absence_of_the_base_bleu_is_rejected_when_the_base_row_was_retrieved():
    s = "The retrieved passages do not state the development BLEU of the base configuration."
    assert tables.absence_contradicted(s, [T3]) == ["the retrieved row 'base' reports bleu"]
    without_base = dict(T3, rows=T3["rows"][1:])
    assert tables.absence_contradicted(s, [without_base]) == []  # true when the row was not retrieved


def test_absence_needs_the_whole_metric_name():
    assert tables.absence_contradicted("The retrieved passages do not state the training time of the big Transformer.",
                                       [BIG]) == []
    assert tables.absence_contradicted("The retrieved passages do not state the training cost of the big Transformer.",
                                       [BIG]) == ["the retrieved row 'Transformer (big)' reports training cost"]
