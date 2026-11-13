"""LaTeX normalization for the NLI input (policy v3): equivalent spellings meet, meaning survives."""
from relay.verify.mathnorm import plain_math

# q05: the same formula, spelled by the passage and by the two real drafts
PASSAGE = (r"p_t = \frac{1}{2} \left(1 - \frac{q_t^\top k_{t-1}}{\left\Vert q_t \right\Vert "
           r"\left\Vert k_{t-1} \right\Vert}\right) \in [0, 1]")
DRAFT1 = r"\(p_t = \tfrac12\bigl(1 - \frac{q_t^\top k_{t-1}}{\|q_t\|\|k_{t-1}\|}\bigr)\) which lies in \([0,1]\)"
DRAFT2 = r"\(p_t = \tfrac12\!\left(1 - \frac{q_t^\top k_{t-1}}{\|q_t\|\|k_{t-1}\|}\right) \in [0,1]\)"


def test_q05_spellings_of_one_formula_become_the_same_text():
    canon = plain_math(PASSAGE)
    assert canon == "p_t = 1 / 2 (1 - (q_t^T k_(t - 1)) / (|| q_t || || k_(t - 1) ||)) in [0, 1]"
    assert plain_math(DRAFT2) == canon
    assert plain_math(DRAFT1).startswith(canon.split(" in ")[0])  # same formula; the prose after it differs


def test_meaning_is_kept_accents_numbers_operators():
    assert plain_math(r"\(\hat{x}_t\)") == "x-hat_t" != plain_math(r"\(x_t\)")  # x-hat is not x
    assert plain_math("x̂_t") == "x-hat_t"                                    # the parser's unicode form
    assert plain_math(r"\(\bar{z}_t = P_t \hat{z}_t + (1-P_t)\,\bar{z}_{t-1}\)") == \
        "z-bar_t = P_t z-hat_t + (1 - P_t) z-bar_(t - 1)"
    assert plain_math(r"\(b_t = \mathds{1}_{\{p_t \geq 0.5\}}\)") == "b_t = 1_(p_t >= 0.5)"
    assert plain_math(r"\(\sin(pos / 10000^{2i/d_{\text{model}}})\)") != plain_math(r"\(\cos(pos / 10000^{2i/d_{\text{model}}})\)")
    assert plain_math(r"\(\tfrac{1}{L}\)") == plain_math(r"\(\frac{1}{L}\)") == "1 / L"


def test_prose_unicode_and_line_breaks():
    assert plain_math("row A, h = 1, single‑head") == "row A, h = 1, single-head"
    assert plain_math("H-Net uses 4,500 low-rank steps.\nNext line.") == "H-Net uses 4,500 low-rank steps.\nNext line."


def test_subscript_groups_keep_variable_identity():
    assert plain_math(r"\(k_{t-1}\)") == "k_(t - 1)" != plain_math(r"\(k_t - 1\)")
    assert plain_math(r"\(d_{\text{model}}^{-0.5}\)") == plain_math(r"\(d_{model}^{-0.5}\)") == "d_model^(- 0.5)"
