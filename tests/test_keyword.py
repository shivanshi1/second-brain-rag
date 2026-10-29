from secondbrain.retrieval.keyword import _or_terms


def test_or_terms_joins_plain_words_without_preserving_query_operators():
    assert _or_terms("When should I use a monolith?") == "when | should | i | use | a | monolith"


def test_or_terms_ignores_punctuation_only_queries():
    assert _or_terms("?! -- ()") == ""
