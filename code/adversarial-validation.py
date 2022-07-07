import pandas as pd
import numpy as np

from typing import Dict, Callable

from sklearn.model_selection import train_test_split

import shap



class AdversarialValidation:

	"""

	"""

	def __init__(self,
	  			 features:list = [],
	  			 base_estimator:Callable = None,
	  			 time_column:str = None,
	  			 train_end_date:str = None,
	  			 random_state:int = 42
	  			 ):
        
		self.features = features
		self.base_estimator = base_estimator
		self.time_column = time_column
		self.train_end_date = train_end_date
		self.random_state = random_state

	def _make_split(self, frame):

		frame.loc[:,"target"] = np.where(frame[self.time_column] < self.train_end_date, 0, 1)

		X_train, X_train, y_train, y_test = train_test_split(frame[self.features],
		 													 frame["target"],
		 													 test_size = self.test_size,
		 													 random_state = self.random_state
		 													)

		return X_train, X_test, y_train, y_test


	def fit(self,frame:pd.DataFrame):
		"""

		"""

		X_train, X_test, y_train, y_test = self._make_split(frame)



		if self.base_estimator is None:
			from lightgbm import LGBMClassfier
			self.base_estimator = LGBMClassfier(**estimator_params).fit(self.X_train, self.y_train)

		predictions = self.predict_proba(X_test)[:,1]

		self.performance_ = roc_auc_score(y_true = y_test, y_score = predictions)

		return self

	def predict(self, frame):

		predictions = self.base_estimator.predict(frame[self.features])

		return predictions


	def predict_proba(self, frame):

		predictions = self.base_estimator.predict_proba(frame[self.features])

		return predictions


	def plot_shap_values(self,frame, plot_type:str = "dot"):
		
		"""

		"""

		_, X, _, _ = self._make_split(frame)

		explainer = shap.TreeExplainer(model = self.base_estimator)
		shap_values = explainer.shap_values(X)
		shap.summary_plot(shap_values, feature_names = self.features, plot_type = plot_type)













