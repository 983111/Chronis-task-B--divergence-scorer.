from run import classify

def test_blind_spot():
    assert classify({
        "behaviors":["a","b","c"],
        "self_talk":[]
    }) == "blind_spot"

def test_aspiration_gap():
    assert classify({
        "behaviors":["a"],
        "self_talk":["x","y","z"]
    }) == "insufficient_evidence"

def test_insufficient():
    assert classify({
        "behaviors":["a"],
        "self_talk":["x"]
    }) == "insufficient_evidence"
