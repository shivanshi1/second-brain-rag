from secondbrain.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_wins():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]])
    assert fused[0][0] in (1, 3)
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_scores_are_descending():
    scores = [s for _, s in reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])]
    assert scores == sorted(scores, reverse=True)


def test_empty_input():
    assert reciprocal_rank_fusion([[], []]) == []
