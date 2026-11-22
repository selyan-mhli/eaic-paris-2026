import t2lib
def test_mcc_hand():
    # practice : isoler exactement ref-04 et ref-07 doit donner 0,524 (valeur du kit officiel, audit §P2)
    truth = {"ref-01":"S","ref-03":"S","ref-05":"S","ref-06":"S","ref-08":"S","ref-02":"T1","ref-04":"T3","ref-07":"T4","ref-09":"T2"}
    pred = {e: ("X" if e not in ("ref-04","ref-07") else e) for e in truth}
    m, cnt = t2lib.pair_mcc(truth, pred)
    assert cnt == (10, 11, 0, 15) and abs(m - 0.524) < 1e-3
def test_constant_is_zero():
    truth = {"a":1,"b":1,"c":2}
    assert t2lib.pair_mcc(truth, {"a":0,"b":0,"c":0})[0] == 0.0
    assert t2lib.pair_mcc(truth, {"a":0,"b":1,"c":2})[0] == 0.0
def test_server_score_perfect_and_cross_family():
    tf = {"a":"F","b":"F","c":"G","d":"G"}; tm = {"a":"1","b":"1","c":"2","d":"3"}
    s, mf, mm, _ = t2lib.server_score(tf, tm, tf, tm); assert s == 1.0
    # même label modèle mais familles prédites différentes => « différents »
    s2 = t2lib.server_score(tf, tm, {"a":"F","b":"H","c":"G","d":"G"}, tm)
    assert s2[3][2] == 1  # FN sur (a,b)
def test_rules_return_partition():
    eps = ["x%d" % i for i in range(6)]
    D = {}
    for a in eps:
        for b in eps:
            if a != b: D[a, b] = 0.1 if "x5" not in (a, b) else 0.9
    iso = t2lib.rule_isolate(eps, D, 2.0); assert iso["x5"] == "x5" and iso["x0"] == "BIG"
    lk = t2lib.rule_linkage(eps, D, 0.5); assert lk["x0"] == lk["x1"] != lk["x5"]
