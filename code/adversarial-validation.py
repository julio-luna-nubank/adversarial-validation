import pandas as pd
import numpy as np

from copy import deepcopy

from typing import Dict, Callable

from sklearn.model_selection import train_test_split

import shap

import tqdm

from xgboost import XGBClassfier


class AdversarialValidation:

    """ """

    def __init__(
        self,
        features: list = [],
        base_estimator: Callable = None,
        time_column: str = None,
        train_end_date: str = None,
        random_state: int = 42,
    ):

        self.features = features
        self.base_estimator = base_estimator
        self.time_column = time_column
        self.train_end_date = train_end_date
        self.random_state = random_state

    def _make_split(self, frame):

        frame.loc[:, "target"] = np.where(
            frame[self.time_column] < self.train_end_date, 0, 1
        )

        X_train, X_train, y_train, y_test = train_test_split(
            frame[self.features],
            frame["target"],
            test_size=self.test_size,
            random_state=self.random_state,
        )

        return X_train, X_test, y_train, y_test

    def fit(self, frame: pd.DataFrame = pd.DataFrame([]), estimator_params: dict = {}):
        """ """

        X_train, X_test, y_train, y_test = self._make_split(frame)

        if self.base_estimator is None:
            self.base_estimator = XGBClassfier(**estimator_params).fit(
                self.X_train, self.y_train
            )

        predictions = self.predict_proba(X_test)[:, 1]

        self.performance_ = roc_auc_score(y_true=y_test, y_score=predictions)

        explainer = shap.TreeExplainer(model=self.base_estimator)
        self.shap_values = explainer.shap_values(X_test)

        return self

    def predict(self, frame):

        predictions = self.base_estimator.predict(frame[self.features])

        return predictions

    def predict_proba(self, frame):

        predictions = self.base_estimator.predict_proba(frame[self.features])

        return predictions

    def plot_shap_values(self, frame, plot_type: str = "dot"):

        """ """

        shap.summary_plot(
            self.shap_values, feature_names=self.features, plot_type=plot_type
        )

    def recursive_feature_elimination(
        self,
        frame: pd.DataFrame,
        n_features_remove: int = None,
        threshold_remove_until: float = 0.5,
    ):

        """ """

        ldf = list()
        tmp_performance: float = 1.0

        all_features_shap = pd.DataFrame(
            {"average_shap_value": np.abs(self.shap_values).mean()}, index=self.features
        ).sort_values(ascending=False)
        all_features_adversarial_performance = self.performance_
        most_important_feature: str = all_features_shap.idxmax()

        ldf.append(("all_features", all_features_adversarial_performance))

        tmp_features = self.features[:]

        if n_features_remove is not None:
            for iter in tqdm(range(n_features_remove)):
                tmp_frame = deepcopy(frame)
                tmp_frame.drop(most_important_feature, axis=1, inplace=True)
                tmp_features.remove(most_important_feature)
                tmp_estimator = self.fit(tmp_frame)

                tmp_features_shap = pd.DataFrame(
                    {"average_shap_value": np.abs(self.shap_values).mean()},
                    index=tmp_features,
                ).sort_values(ascending=False)
                most_important_feature: str = all_features_shap.idxmax()
                tmp_performance = self.performance_

                ldf.append((most_important_feature, tmp_performance))
        elif threshold_remove_until is not None:
            while (tmp_performance > threshold_remove_until) and len(tmp_features) > 0:
                tmp_frame = deepcopy(frame)
                tmp_frame.drop(most_important_feature, axis=1, inplace=True)
                tmp_features.remove(most_important_feature)
                tmp_estimator = self.fit(tmp_frame)

                tmp_features_shap = pd.DataFrame(
                    {"average_shap_value": np.abs(self.shap_values).mean()},
                    index=tmp_features,
                ).sort_values(ascending=False)
                most_important_feature: str = all_features_shap.idxmax()
                tmp_performance = self.performance_

                ldf.append((most_important_feature, tmp_performance))

        return pd.DataFrame(ldf, columns=["columns", "adversarial_model_performance"])
