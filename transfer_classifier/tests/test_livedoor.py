import pandas as pd
from transfer_classifier.dataset_preprocessor.livedoor import Livedoor


class TestLivedoor:
    def test_load(self) -> None:
        livedoor = Livedoor(
            input_column="title",
            label_column="labels",
            lang="ja",
            validation_size=0.2,
            test_size=0.3,
        )

        dataset_root = livedoor.save(force=True)
        assert dataset_root.joinpath("dataset.csv").exists()
        all = pd.read_csv(dataset_root.joinpath("dataset.csv"))

        validation_size = int(len(all) * 0.2)
        test_size = int(len(all) * 0.3)
        train_size = len(all) - validation_size - test_size

        train = livedoor.load("train")
        assert len(train) == train_size
        assert len(livedoor.load("validation")) == validation_size
        assert len(livedoor.load("test")) == test_size

        assert all.url.nunique() == len(all)

        assert pd.read_csv(dataset_root.joinpath("train.csv")).url.nunique() == train_size
        assert pd.read_csv(dataset_root.joinpath("validation.csv")).url.nunique() == validation_size
        assert pd.read_csv(dataset_root.joinpath("test.csv")).url.nunique() == test_size

        assert all.labels.nunique() == 9
        assert all.label_name.nunique() == 9
        assert len(livedoor.load("test")[0]["title"]) > 0
        assert len(livedoor.load("test")[0]["body"]) > 0
        assert livedoor.load("test")[0]["url"].startswith("http")
