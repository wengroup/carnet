from carten.symbolic.linearly_independent import (
    get_E,
    get_E_rules,
    get_G_even,
    get_G_odd,
)


def test_get_E_rules():
    def as_set(data):
        new_data = []
        for d in data:
            new_data.append(frozenset({k: frozenset(v) for k, v in d.items()}.items()))
        return set(new_data)

    out = get_E_rules(1, 0)
    assert as_set(out) == as_set([{"d_rs": ["aA"], "d_rr": [], "d_ss": []}])

    out = get_E_rules(2, 0)
    assert as_set(out) == as_set(
        [
            {"d_rs": ["aA", "bB"], "d_rr": [], "d_ss": []},
            {"d_rs": ["bA", "aB"], "d_rr": [], "d_ss": []},
        ]
    )

    out = get_E_rules(2, 1)
    assert as_set(out) == as_set(
        [
            {"d_rs": [], "d_rr": ["ab"], "d_ss": ["AB"]},
        ]
    )

    out = get_E_rules(3, 0)
    assert as_set(out) == as_set(
        [
            {"d_rs": ["aA", "bB", "cC"], "d_rr": [], "d_ss": []},
            {"d_rs": ["aA", "bC", "cB"], "d_rr": [], "d_ss": []},
            {"d_rs": ["aB", "bA", "cC"], "d_rr": [], "d_ss": []},
            {"d_rs": ["aB", "bC", "cA"], "d_rr": [], "d_ss": []},
            {"d_rs": ["aC", "bA", "cB"], "d_rr": [], "d_ss": []},
            {"d_rs": ["aC", "bB", "cA"], "d_rr": [], "d_ss": []},
        ]
    )

    out = get_E_rules(3, 1)
    assert as_set(out) == as_set(
        [
            {"d_rs": ["cC"], "d_rr": ["ab"], "d_ss": ["AB"]},
            {"d_rs": ["cB"], "d_rr": ["ab"], "d_ss": ["AC"]},
            {"d_rs": ["cA"], "d_rr": ["ab"], "d_ss": ["BC"]},
            {"d_rs": ["bC"], "d_rr": ["ac"], "d_ss": ["AB"]},
            {"d_rs": ["bB"], "d_rr": ["ac"], "d_ss": ["AC"]},
            {"d_rs": ["bA"], "d_rr": ["ac"], "d_ss": ["BC"]},
            {"d_rs": ["aC"], "d_rr": ["bc"], "d_ss": ["AB"]},
            {"d_rs": ["aB"], "d_rr": ["bc"], "d_ss": ["AC"]},
            {"d_rs": ["aA"], "d_rr": ["bc"], "d_ss": ["BC"]},
        ]
    )


def test_E():
    e = get_E(1)
    assert e.to_str_list() == ["+1 δ_aA"]

    e = get_E(2)
    assert set(e.to_str_list()) == {
        "+1/2 δ_aA δ_bB",
        "+1/2 δ_aB δ_bA",
        "-1/3 δ_ab δ_AB",
    }

    e = get_E(3)
    assert set(e.to_str_list()) == {
        "+1/6 δ_aA δ_bB δ_cC",
        "+1/6 δ_aA δ_bC δ_cB",
        "+1/6 δ_aB δ_bA δ_cC",
        "+1/6 δ_aB δ_bC δ_cA",
        "+1/6 δ_aC δ_bA δ_cB",
        "+1/6 δ_aC δ_bB δ_cA",
        "-1/15 δ_cC δ_ab δ_AB",
        "-1/15 δ_cB δ_ab δ_AC",
        "-1/15 δ_cA δ_ab δ_BC",
        "-1/15 δ_bC δ_ac δ_AB",
        "-1/15 δ_bB δ_ac δ_AC",
        "-1/15 δ_bA δ_ac δ_BC",
        "-1/15 δ_aC δ_bc δ_AB",
        "-1/15 δ_aB δ_bc δ_AC",
        "-1/15 δ_aA δ_bc δ_BC",
    }

    e = get_E(4)
    assert set(e.to_str_list()) == {
        ## t = 0
        "+1/24 δ_aA δ_bB δ_cC δ_dD",
        "+1/24 δ_aA δ_bB δ_cD δ_dC",
        "+1/24 δ_aA δ_bC δ_cB δ_dD",
        "+1/24 δ_aA δ_bC δ_cD δ_dB",
        "+1/24 δ_aA δ_bD δ_cB δ_dC",
        "+1/24 δ_aA δ_bD δ_cC δ_dB",
        #
        "+1/24 δ_aB δ_bA δ_cC δ_dD",
        "+1/24 δ_aB δ_bA δ_cD δ_dC",
        "+1/24 δ_aB δ_bC δ_cA δ_dD",
        "+1/24 δ_aB δ_bC δ_cD δ_dA",
        "+1/24 δ_aB δ_bD δ_cA δ_dC",
        "+1/24 δ_aB δ_bD δ_cC δ_dA",
        #
        "+1/24 δ_aC δ_bA δ_cB δ_dD",
        "+1/24 δ_aC δ_bA δ_cD δ_dB",
        "+1/24 δ_aC δ_bB δ_cA δ_dD",
        "+1/24 δ_aC δ_bB δ_cD δ_dA",
        "+1/24 δ_aC δ_bD δ_cA δ_dB",
        "+1/24 δ_aC δ_bD δ_cB δ_dA",
        #
        "+1/24 δ_aD δ_bA δ_cB δ_dC",
        "+1/24 δ_aD δ_bA δ_cC δ_dB",
        "+1/24 δ_aD δ_bB δ_cA δ_dC",
        "+1/24 δ_aD δ_bB δ_cC δ_dA",
        "+1/24 δ_aD δ_bC δ_cA δ_dB",
        "+1/24 δ_aD δ_bC δ_cB δ_dA",
        ## t = 1
        "-1/84 δ_cC δ_dD δ_ab δ_AB",
        "-1/84 δ_cD δ_dC δ_ab δ_AB",
        "-1/84 δ_cB δ_dD δ_ab δ_AC",
        "-1/84 δ_cD δ_dB δ_ab δ_AC",
        "-1/84 δ_cB δ_dC δ_ab δ_AD",
        "-1/84 δ_cC δ_dB δ_ab δ_AD",
        "-1/84 δ_cA δ_dD δ_ab δ_BC",
        "-1/84 δ_cD δ_dA δ_ab δ_BC",
        "-1/84 δ_cA δ_dC δ_ab δ_BD",
        "-1/84 δ_cC δ_dA δ_ab δ_BD",
        "-1/84 δ_cA δ_dB δ_ab δ_CD",
        "-1/84 δ_cB δ_dA δ_ab δ_CD",
        #
        "-1/84 δ_bC δ_dD δ_ac δ_AB",
        "-1/84 δ_bD δ_dC δ_ac δ_AB",
        "-1/84 δ_bB δ_dD δ_ac δ_AC",
        "-1/84 δ_bD δ_dB δ_ac δ_AC",
        "-1/84 δ_bB δ_dC δ_ac δ_AD",
        "-1/84 δ_bC δ_dB δ_ac δ_AD",
        "-1/84 δ_bA δ_dD δ_ac δ_BC",
        "-1/84 δ_bD δ_dA δ_ac δ_BC",
        "-1/84 δ_bA δ_dC δ_ac δ_BD",
        "-1/84 δ_bC δ_dA δ_ac δ_BD",
        "-1/84 δ_bA δ_dB δ_ac δ_CD",
        "-1/84 δ_bB δ_dA δ_ac δ_CD",
        #
        "-1/84 δ_bC δ_cD δ_ad δ_AB",
        "-1/84 δ_bD δ_cC δ_ad δ_AB",
        "-1/84 δ_bB δ_cD δ_ad δ_AC",
        "-1/84 δ_bD δ_cB δ_ad δ_AC",
        "-1/84 δ_bB δ_cC δ_ad δ_AD",
        "-1/84 δ_bC δ_cB δ_ad δ_AD",
        "-1/84 δ_bA δ_cD δ_ad δ_BC",
        "-1/84 δ_bD δ_cA δ_ad δ_BC",
        "-1/84 δ_bA δ_cC δ_ad δ_BD",
        "-1/84 δ_bC δ_cA δ_ad δ_BD",
        "-1/84 δ_bA δ_cB δ_ad δ_CD",
        "-1/84 δ_bB δ_cA δ_ad δ_CD",
        #
        "-1/84 δ_aC δ_dD δ_bc δ_AB",
        "-1/84 δ_aD δ_dC δ_bc δ_AB",
        "-1/84 δ_aB δ_dD δ_bc δ_AC",
        "-1/84 δ_aD δ_dB δ_bc δ_AC",
        "-1/84 δ_aB δ_dC δ_bc δ_AD",
        "-1/84 δ_aC δ_dB δ_bc δ_AD",
        "-1/84 δ_aA δ_dD δ_bc δ_BC",
        "-1/84 δ_aD δ_dA δ_bc δ_BC",
        "-1/84 δ_aA δ_dC δ_bc δ_BD",
        "-1/84 δ_aC δ_dA δ_bc δ_BD",
        "-1/84 δ_aA δ_dB δ_bc δ_CD",
        "-1/84 δ_aB δ_dA δ_bc δ_CD",
        #
        "-1/84 δ_aC δ_cD δ_bd δ_AB",
        "-1/84 δ_aD δ_cC δ_bd δ_AB",
        "-1/84 δ_aB δ_cD δ_bd δ_AC",
        "-1/84 δ_aD δ_cB δ_bd δ_AC",
        "-1/84 δ_aB δ_cC δ_bd δ_AD",
        "-1/84 δ_aC δ_cB δ_bd δ_AD",
        "-1/84 δ_aA δ_cD δ_bd δ_BC",
        "-1/84 δ_aD δ_cA δ_bd δ_BC",
        "-1/84 δ_aA δ_cC δ_bd δ_BD",
        "-1/84 δ_aC δ_cA δ_bd δ_BD",
        "-1/84 δ_aA δ_cB δ_bd δ_CD",
        "-1/84 δ_aB δ_cA δ_bd δ_CD",
        #
        "-1/84 δ_aC δ_bD δ_cd δ_AB",
        "-1/84 δ_aD δ_bC δ_cd δ_AB",
        "-1/84 δ_aB δ_bD δ_cd δ_AC",
        "-1/84 δ_aD δ_bB δ_cd δ_AC",
        "-1/84 δ_aB δ_bC δ_cd δ_AD",
        "-1/84 δ_aC δ_bB δ_cd δ_AD",
        "-1/84 δ_aA δ_bD δ_cd δ_BC",
        "-1/84 δ_aD δ_bA δ_cd δ_BC",
        "-1/84 δ_aA δ_bC δ_cd δ_BD",
        "-1/84 δ_aC δ_bA δ_cd δ_BD",
        "-1/84 δ_aA δ_bB δ_cd δ_CD",
        "-1/84 δ_aB δ_bA δ_cd δ_CD",
        # t = 3
        "+1/105 δ_ab δ_cd δ_AB δ_CD",
        "+1/105 δ_ab δ_cd δ_AC δ_BD",
        "+1/105 δ_ab δ_cd δ_AD δ_BC",
        #
        "+1/105 δ_ac δ_bd δ_AB δ_CD",
        "+1/105 δ_ac δ_bd δ_AC δ_BD",
        "+1/105 δ_ac δ_bd δ_AD δ_BC",
        #
        "+1/105 δ_ad δ_bc δ_AB δ_CD",
        "+1/105 δ_ad δ_bc δ_AC δ_BD",
        "+1/105 δ_ad δ_bc δ_AD δ_BC",
    }


def test_G_even():
    # n=0, j=0
    all_G = get_G_even(j=0, n=0)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {"+1"}

    # n=1, j=1
    all_G = get_G_even(j=1, n=1)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {"+1 δ_aA"}

    # n=2, j=0
    all_G = get_G_even(j=0, n=2)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {"+1 δ_AB"}

    # n=2, j=2
    all_G = get_G_even(j=2, n=2)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == {
        "-1/3 δ_ab δ_AB",
        "+1/2 δ_aB δ_bA",
        "+1/2 δ_aA δ_bB",
    }

    # n=3, j=1
    all_G = get_G_even(j=1, n=3)
    assert len(all_G) == 3
    assert set(all_G[0].to_str_list()) == {"+1 δ_aA δ_BC"}
    assert set(all_G[1].to_str_list()) == {"+1 δ_aB δ_AC"}
    assert set(all_G[2].to_str_list()) == {"+1 δ_aC δ_AB"}

    # n=3, j=3
    all_G = get_G_even(j=3, n=3)
    assert len(all_G) == 1
    assert set(all_G[0].to_str_list()) == set(get_E(3).to_str_list())

    # n=4, j=0
    all_G = get_G_even(j=0, n=4)
    assert len(all_G) == 3
    assert set(all_G[0].to_str_list()) == {"+1 δ_AB δ_CD"}
    assert set(all_G[1].to_str_list()) == {"+1 δ_AC δ_BD"}
    assert set(all_G[2].to_str_list()) == {"+1 δ_AD δ_BC"}

    # n=4, j=2
    all_G = get_G_even(j=2, n=4)
    assert len(all_G) == 6
    assert set(all_G[0].to_str_list()) == {
        "+1/2 δ_aA δ_bB δ_CD",
        "+1/2 δ_aB δ_bA δ_CD",
        "-1/3 δ_ab δ_AB δ_CD",
    }
    assert set(all_G[1].to_str_list()) == {
        "+1/2 δ_aA δ_bC δ_BD",
        "+1/2 δ_aC δ_bA δ_BD",
        "-1/3 δ_ab δ_AC δ_BD",
    }

    assert set(all_G[2].to_str_list()) == {
        "+1/2 δ_aA δ_bD δ_BC",
        "+1/2 δ_aD δ_bA δ_BC",
        "-1/3 δ_ab δ_AD δ_BC",
    }

    assert set(all_G[3].to_str_list()) == {
        "+1/2 δ_aB δ_bC δ_AD",
        "+1/2 δ_aC δ_bB δ_AD",
        "-1/3 δ_ab δ_BC δ_AD",
    }

    assert set(all_G[4].to_str_list()) == {
        "+1/2 δ_aB δ_bD δ_AC",
        "+1/2 δ_aD δ_bB δ_AC",
        "-1/3 δ_ab δ_BD δ_AC",
    }

    assert set(all_G[5].to_str_list()) == {
        "+1/2 δ_aC δ_bD δ_AB",
        "+1/2 δ_aD δ_bC δ_AB",
        "-1/3 δ_ab δ_CD δ_AB",
    }


def test_G_odd():
    # n = 1, j = 0 not possible

    # # n = 2, j = 1
    # all_G = get_G_odd(j=1, n=2)
    # assert len(all_G) == 1
    # assert set(all_G[0].to_str_list()) == {"(1) δ_aC ε_CAB"}

    # TODO, seems we need to implement triple products os epsilon
    # n 3, j = 0
    # all_G = get_G_odd(j=0, n=3)
    # assert len(all_G) == 3
    # assert set(all_G[0].to_str_list()) == {"(1) ε_CAB δ_aC"}

    # n = 3, j = 2
    all_G = get_G_odd(j=2, n=3)
    assert len(all_G) == 3
    assert set(all_G[0].to_str_list()) == {
        "-1/3 δ_ab δ_CD ε_DAB",
        "+1/2 δ_aD δ_bC ε_DAB",
        "+1/2 δ_aC δ_bD ε_DAB",
    }
    assert set(all_G[1].to_str_list()) == {
        "-1/3 δ_ab δ_BD ε_DAC",
        "+1/2 δ_aD δ_bB ε_DAC",
        "+1/2 δ_aB δ_bD ε_DAC",
    }
    assert set(all_G[2].to_str_list()) == {
        "-1/3 δ_ab δ_AD ε_DBC",
        "+1/2 δ_aD δ_bA ε_DBC",
        "+1/2 δ_aA δ_bD ε_DBC",
    }

    # n = 4, j = 1
    all_G = get_G_odd(j=1, n=4)
    assert len(all_G) == 6
    assert set(all_G[0].to_str_list()) == {"+1 δ_aE ε_EAB δ_CD"}
    assert set(all_G[1].to_str_list()) == {"+1 δ_aE ε_EAC δ_BD"}
    assert set(all_G[2].to_str_list()) == {"+1 δ_aE ε_EAD δ_BC"}
    assert set(all_G[3].to_str_list()) == {"+1 δ_aE ε_EBC δ_AD"}
    assert set(all_G[4].to_str_list()) == {"+1 δ_aE ε_EBD δ_AC"}
    assert set(all_G[5].to_str_list()) == {"+1 δ_aE ε_ECD δ_AB"}

    # n = 4, j = 3
    all_G = get_G_odd(j=3, n=4)
    assert len(all_G) == 6
    assert set(all_G[0].to_str_list()) == {
        "-1/15 δ_aC δ_bc δ_DE ε_EAB",
        "-1/15 δ_aD δ_bc δ_CE ε_EAB",
        "-1/15 δ_aE δ_bc δ_CD ε_EAB",
        "-1/15 δ_bC δ_ac δ_DE ε_EAB",
        "-1/15 δ_bD δ_ac δ_CE ε_EAB",
        "-1/15 δ_bE δ_ac δ_CD ε_EAB",
        "-1/15 δ_cC δ_ab δ_DE ε_EAB",
        "-1/15 δ_cD δ_ab δ_CE ε_EAB",
        "-1/15 δ_cE δ_ab δ_CD ε_EAB",
        "+1/6 δ_aC δ_bD δ_cE ε_EAB",
        "+1/6 δ_aC δ_bE δ_cD ε_EAB",
        "+1/6 δ_aD δ_bC δ_cE ε_EAB",
        "+1/6 δ_aD δ_bE δ_cC ε_EAB",
        "+1/6 δ_aE δ_bC δ_cD ε_EAB",
        "+1/6 δ_aE δ_bD δ_cC ε_EAB",
    }
    # ignore a couple of them
    assert set(all_G[5].to_str_list()) == {
        "-1/15 δ_aA δ_bc δ_BE ε_ECD",
        "-1/15 δ_aB δ_bc δ_AE ε_ECD",
        "-1/15 δ_aE δ_bc δ_AB ε_ECD",
        "-1/15 δ_bA δ_ac δ_BE ε_ECD",
        "-1/15 δ_bB δ_ac δ_AE ε_ECD",
        "-1/15 δ_bE δ_ac δ_AB ε_ECD",
        "-1/15 δ_cA δ_ab δ_BE ε_ECD",
        "-1/15 δ_cB δ_ab δ_AE ε_ECD",
        "-1/15 δ_cE δ_ab δ_AB ε_ECD",
        "+1/6 δ_aA δ_bB δ_cE ε_ECD",
        "+1/6 δ_aA δ_bE δ_cB ε_ECD",
        "+1/6 δ_aB δ_bA δ_cE ε_ECD",
        "+1/6 δ_aB δ_bE δ_cA ε_ECD",
        "+1/6 δ_aE δ_bA δ_cB ε_ECD",
        "+1/6 δ_aE δ_bB δ_cA ε_ECD",
    }
