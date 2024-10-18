from carten.utils import multi_double_index


def test_multi_double_index():
    assert multi_double_index(2) == ["ab", "cd"]
    assert multi_double_index(3, start=1) == ["bc", "de", "fg"]
