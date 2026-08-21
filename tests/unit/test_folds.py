from water_agent.vision.folds import assign_validation_folds, group_ids_from_audit


def test_grouped_folds_never_split_a_group() -> None:
    filenames = [f"{index}.jpg" for index in range(8)]
    groups = group_ids_from_audit(filenames, [["0.jpg", "1.jpg"], ["4.jpg", "5.jpg"]])
    labels = ["A", "A", "A", "A", "B", "B", "B", "B"]
    folds = assign_validation_folds(labels, groups, n_splits=2, seed=7)
    assert folds[0] == folds[1]
    assert folds[4] == folds[5]

