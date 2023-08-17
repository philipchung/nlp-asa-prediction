from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import numpy as np


@dataclass
class ModelPredictions(dict):
    asa_class_names: Sequence | None = None
    asa_label2id: Sequence | None = None
    labels: np.ndarray | None = None
    rf_preds: np.ndarray | None = None
    rf_pred_proba: np.ndarray | None = None
    svm_preds: np.ndarray | None = None
    svm_pred_proba: np.ndarray | None = None
    fasttext_preds: np.ndarray | None = None
    fasttext_pred_proba: np.ndarray | None = None
    bioclinicalbert_preds: np.ndarray | None = None
    bioclinicalbert_pred_proba: np.ndarray | None = None
    age_meds_preds: np.ndarray | None = None
    age_meds_pred_proba: np.ndarray | None = None
    random_classifier_preds: np.ndarray | None = None
    random_classifier_pred_proba: np.ndarray | None = None

    def __setitem__(self, key, value) -> None:
        self.__dict__[key] = value

    def __getitem__(self, key) -> Any:
        return self.__dict__[key]

    def __delitem__(self, key) -> None:
        del self.__dict__[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self) -> str:
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self) -> str:
        """echoes class, id, & reproducible representation in the REPL"""
        return "{}, ModelPredictions({})".format(
            super(ModelPredictions, self).__repr__(), self.__dict__
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricResults(dict):
    rf: dict | None = None
    svm: dict | None = None
    fasttext: dict | None = None
    bioclinicalbert: dict | None = None
    age_meds: dict | None = None
    random_classifier: dict | None = None

    def __setitem__(self, key, value) -> None:
        self.__dict__[key] = value

    def __getitem__(self, key) -> Any:
        return self.__dict__[key]

    def __delitem__(self, key) -> None:
        del self.__dict__[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self) -> str:
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self) -> str:
        """echoes class, id, & reproducible representation in the REPL"""
        return "{}, MetricResults({})".format(
            super(MetricResults, self).__repr__(), self.__dict__
        )

    def dict(self) -> dict[str, Any]:
        return asdict(self)
