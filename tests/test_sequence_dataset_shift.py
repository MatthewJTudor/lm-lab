from lm_lab.data.sequence_dataset import SequenceDataset, SequenceDatasetConfig


def test_sequence_dataset_shift() -> None:
    tokens = [0, 1, 2, 3, 4, 5]
    cfg = SequenceDatasetConfig(block_size=3)

    ds = SequenceDataset(tokens, cfg)

    assert len(ds) == 3  # 6 - 3

    x0, y0 = ds[0]
    assert list(x0) == [0, 1, 2]
    assert list(y0) == [1, 2, 3]

    x1, y1 = ds[1]
    assert list(x1) == [1, 2, 3]
    assert list(y1) == [2, 3, 4]