from ml import data as D


def test_checksum_matches_source_md():
    assert D.sha256_file(D.DATA_PATH) == D.EXPECTED_SHA256
    source = (D.PROJECT_ROOT / "data" / "SOURCE.md").read_text(encoding="utf-8")
    assert D.EXPECTED_SHA256 in source


def test_rows_and_columns():
    df = D.load_raw()
    assert len(df) == D.EXPECTED_ROWS
    assert list(df.columns) == [D.RENAME_MAP[c] for c in D.RAW_COLUMNS]


def test_original_header():
    with open(D.DATA_PATH, encoding="utf-8-sig") as handle:
        header = handle.readline().strip().split(",")
    assert header == D.RAW_COLUMNS


def test_split_is_stratified_and_disjoint():
    df = D.load_raw()
    train, test = D.split_train_test(df)
    assert len(train) == 8000 and len(test) == 2000
    assert set(train["udi"]).isdisjoint(test["udi"])
    assert abs(train[D.TARGET].mean() - test[D.TARGET].mean()) < 0.002


def test_published_rules_reproduce_flags():
    df = D.load_raw()
    agreement = D.label_consistency(df)["rule_agreement"]
    assert agreement == {"hdf": 10_000, "pwf": 10_000, "osf": 10_000}
