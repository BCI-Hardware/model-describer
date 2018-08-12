# this is a test file
# sensitivity plot creation and testing

import pandas as pd
import numpy as np
from collections import deque
import warnings

from mdesc.data.datamanager import (DataManager, DataUtilities)
from mdesc.utils.metrics import MetricMixin
from mdesc.utils.mdesc_utils import prob_acc


class Eval(MetricMixin):

    def __init__(self, round_num=2, prediction_fn=None,
                 error_type='MSE', model_type='regression', target_names=None,
                 groupby_names=None, feature_names=None, target_classes=None):

        """

        :param round_num:
        :param prediction_fn:
        :param error_type:
        :param model_type:
        :param target_names: array type
            (optional) names of classes that describe model outputs
        """

        self.round_num = round_num
        self.prediction_fn = prediction_fn
        self.error_type = error_type
        self.data_container = deque()
        self.model_type = model_type
        self.target_names = target_names
        self.groupby_names = groupby_names
        self.feature_names = feature_names
        self.target_classes = target_classes

    def _create_errors(self, X, y,
                       original_preds=None):
        """
        construct error metrics based on difference between y_hat and y_true

        For regression tasks, simply measure the difference between y_hat and y_true.
        For classification tasks, get the predicted probability of falling in class 1.
        Using the predicted probability, get the prediction accuracy against the true class.

        If original_preds specified, measure the difference between new predictions and
        original_predictions to develop sensitivity measures.

        :param X: np.array - required
            Input X array
        :param y: np.array - required
            Input y array
        :param original_preds: np.array - optional
            original prediction array
        :return: np.array
            array of error measures
        """

        def unpack_preds():
            pred_proba_y = self.prediction_fn(X)
            actual_y = [self.data_set.target_classes[x] for x in y]
            prob_diff = [prob_acc(true_class=actual_value, pred_prob=pred_value[1]) for actual_value, pred_value in
                         zip(actual_y, pred_proba_y)]
            return np.array(prob_diff)

        if self.model_type == 'regression':
            y_pred = self.prediction_fn(X)
            if original_preds is not None:
                errors = original_preds - y_pred
            else:
                errors = y - y_pred
        else:
            errors = unpack_preds()
            if original_preds is not None:
                errors = original_preds - errors

        return errors

    def _make_data(self, X, y, groupby_df, **kwargs):

        if isinstance(self, Sensitivity):
            self.class_type = 'sensitivity'
        else:
            self.class_type = 'error'

        self.data_set = DataManager(X=X, y=y, groupby_df=groupby_df,
                                    target_name=self.target_names,
                                    target_classes=self.target_classes,
                                    feature_names=self.feature_names,
                                    groupby_names=self.groupby_names,
                                    model_type=self.model_type,
                                    class_type=self.class_type
                                    )

        kwargs['error_type'] = self.error_type

        groups = self.data_set.create_sub_groups(**kwargs)
        return groups

    def _reset_state(self):
        """create returnable results and reset container"""
        self.data_set.results = pd.concat(self.data_container)
        self.data_container = []

    def construct_group_aggregates(self, group_data, errors=None,
                                   **kwargs):
        """
        summarize group level slice of data

        :param group_data: np.array - required
            slice of original data within specific group
        :param groupby_var: str - required
            string name of groupby level
        :param group_level: str - required
            specific level within groupby_var
        :param errors: np.array - required
            list or np.array of error metric
        :param x_value: float - required
            value of continuous variables (specific percentile)
        :param x_name: str - required
            feature name of X
        :param y_slice: np.array - required
            predicted y values for slice of data
        :param std_change: float - optional
            user defined standard deviation change in X
        """
        # replace nan's with 0 in pos/neg errors
        if self.class_type == 'sensitivity':
            y_slice = np.nanmean(errors)
        else:
            y_slice = np.nanmean(kwargs['y_slice'])

        positive_errors = np.nan_to_num(np.nanmedian(errors[errors >= 0]), 0)
        negative_errors = np.nan_to_num(np.nanmedian(errors[errors <= 0]), 0)

        errdf = pd.DataFrame({'groupByValue': kwargs['group_level'],
                              'groupByVarName': kwargs['groupby_var'],
                              self.error_type: self.metric_all(errors, metric=self.error_type),
                              'Total': float(group_data.shape[0]),
                              'x_name': kwargs['colname'],
                              'x_value': kwargs['percentile_value'],
                              'errPos': positive_errors,
                              'errNeg': negative_errors,
                              'predictedYSmooth': y_slice,
                              'dtype': kwargs['dtype'],
                              'std_change': kwargs.get('std_change', None)}, index=[0])

        self.data_container.append(errdf)

    def fit(self, X, y, groupby_df=None,
            errors=None):
        """
        fit X, y and build aggregate performance stats by region of data

        :param X: np.array or pd.DataFrame - required
            input X used to build model
        :param y: np.array - required
            input y used to build model
        :param groupby_df: np.array or pd.DataFrame - optional
            groupby values to build metrics within regions of data. If left None,
            default is to summarize thte entire dataset with an 'all' indicator
        :param errors: np.array - optional
            modified error values to use for calculation
        """

        if errors is None:
            errors = self._create_errors(X, y)

        groups = self._make_data(X, y, groupby_df, errors=errors)

        for group in groups:
            group['y_slice'] = self.data_set.y[group['percentile_indices']]
            self.construct_group_aggregates(self.data_set.X[group['percentile_indices']],
                                            errors=errors[group['percentile_indices']],
                                            **group)

        self._reset_state()

    def fit_transform(self, X, y, groupby_df=None,
                      **kwargs):
        """
        fit X,y and transform by returning aggregate measures

        :param X: np.array or pd.DataFrame - required
            input X used to build model
        :param y: np.array - required
            input y used to build model
        :param groupby_df: np.array or pd.DataFrame - optional
            groupby values to build metrics within regions of data. If left None,
            default is to summarize thte entire dataset with an 'all' indicator
        :param kwargs: optional kywrd arguments
            errors: np.array - optional
                modified error values to use for calculation
        :return: pd.DataFrame
            transformed dataset
        """

        self.fit(X=X, y=y, groupby_df=groupby_df,
                 **kwargs)

        return self.data_set.results


class Sensitivity(Eval):

    def __init__(self, std=1, **kwargs):
        """


        :param std: int|float - optional
            standard deviation number to construct synthetic data
        :param kwargs:

        Examples
        ---------
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        >>> from mdesc.models import Sensitivity

        >>> modelObjc = RandomForestRegressor()
        >>> # wine quality dataset example
        >>> wine = pd.read_csv('debug/wine.csv')
        >>> groupby_df = wine[['alcohol', 'Type']]
        >>> ydepend = 'quality'
        >>> X = wine.loc[:, wine.columns != ydepend]
        >>> y = wine.loc[:, ydepend]
        >>> X = pd.get_dummies(X)
        >>> modelObjc.fit(X, y)
        >>> # being sensitivity
        >>> RE = Sensitivity(prediction_fn=modelObjc.predict,
        >>>                      model_type='regression')
        >>> res = RE.fit_transform(X=X, y=y)
        >>> res.head()
        MSE  Total groupByValue groupByVarName            x_name  x_value
        0.040000    1.0         high        alcohol  volatile acidity    0.115
        0.172500    4.0         high        alcohol  volatile acidity    0.160
        0.245000    6.0         high        alcohol  volatile acidity    0.170
        """

        if abs(std) > 3:
            warnings.warn("""Standard deviation number set above 3. Unreliable sensitivities
            are likely to occur. std: {}""".format(std))
        self.std = std
        self.data_container = deque()

        super(Sensitivity, self).__init__(**kwargs)

    def _make_error_dict(self):
        """
        Construct lookup dictionary mapping continuous features to errors based on
        sensitivity adjustment (i.e. self.std)

        :return: dict
            {'column_1': [1, 0.9, 1, 0.23, ...]}
        """
        if self.model_type == 'classification':
            original_preds = self._create_errors(self.data_set.X, self.data_set.y)
        else:
            original_preds = self.prediction_fn(self.data_set.X)

        error_dict = {}
        change_dict = {}
        for idx, fname in enumerate(self.data_set.feature_names):
            if not DataUtilities.isbinary(self.data_set.X[:, idx]):
                X_copy = np.copy(self.data_set.X)
                change = np.std(self.data_set.X[:, idx])
                X_copy[:, idx] = X_copy[:, idx] + (change * self.std)
                errors = self._create_errors(X_copy, self.data_set.y,
                                             original_preds=original_preds)
                error_dict[idx] = errors
                change_dict[idx] = change

        return error_dict, change_dict

    def fit(self, X, y, groupby_df=None,
            **kwargs):
        """
        fit X,y and transform by returning aggregate measures

        :param X: np.array or pd.DataFrame - required
            input X used to build model
        :param y: np.array - required
            input y used to build model
        :param groupby_df: np.array or pd.DataFrame - optional
            groupby values to build metrics within regions of data. If left None,
            default is to summarize thte entire dataset with an 'all' indicator
        :param kwargs: optional kywrd arguments
            errors: np.array - optional
                modified error values to use for calculation
        :return: pd.DataFrame
            transformed dataset
        """
        original_errors = self._create_errors(X, y)
        groups = self._make_data(X, y, groupby_df, errors=original_errors)
        error_dict, change_dict = self._make_error_dict()

        for group in groups:
            y_slice = self.data_set.y[group['percentile_indices']]
            group['y_slice'] = y_slice
            col_idx = np.where(self.data_set.feature_names == group['colname'])[0][0]
            group['std_change'] = change_dict[col_idx]
            # col_idx = self.data_set.feature_names.index(col_name)

            self.construct_group_aggregates(self.data_set.X[group['percentile_indices']],
                                            errors=error_dict[col_idx][group['percentile_indices']],
                                            **group)

        self._reset_state()
