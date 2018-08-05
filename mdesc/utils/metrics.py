#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class MetricMixin(object):
    """Mixin for all score based objects"""

    @staticmethod
    def metric_mse(errors):
        """return errors MSE"""

        return np.mean(errors ** 2)

    @staticmethod
    def metric_rmse(errors):
        """return errors RMSE"""

        return (np.mean(errors ** 2)) ** (1 / 2)

    @staticmethod
    def metric_mae(self,
                   errors):
        """return errors mae"""

        return np.sum(np.absolute(errors)) / len(errors)

    @staticmethod
    def metric_mean(errors):
        """return errors mean"""

        return np.mean(errors)

    @staticmethod
    def metric_median(errors):
        """return errors median"""

        return np.median(errors)

    def metric_all(self,
                   errors,
                   metric='MAE'):
        """Switch metric methods based on input"""

        metric_lookup = {'MAE': MetricMixin.metric_mae,
                         'MSE': MetricMixin.metric_mse,
                         'RMSE': MetricMixin.metric_rmse,
                         'MEAN': MetricMixin.metric_mean,
                         'MEDIAN': MetricMixin.metric_median}

        return metric_lookup[metric](errors)


class InsightMixin(MetricMixin):
    """Mixin class for all metric based objects"""

    def create_insights(self,
                        group_data,
                        groupby_var=None,
                        error_type='RMSE'):
        """
        aggregates user specified error metric from raw errors

        :param group: dataframe containing errors
        :param group_var: str specificying groupby variable
        :param error_type: str specifying error metric
        :return error metric dataframe
        :rtype pd.DataFrame
        """

        errors = group_data['errors']

        errdf = pd.DataFrame({'groupByValue': group_data.name,
                              'groupByVarName': groupby_var,
                              error_type: self.metric_all(errors, metric=error_type),
                              'Total': float(group_data.shape[0])}, index=[0])
        return errdf


