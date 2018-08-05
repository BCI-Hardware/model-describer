# !/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
import logging
import six

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

from mdesc.utils import utils as md_utils
from mdesc.utils import check_utils as checks
from mdesc.utils import percentiles
from mdesc.utils import formatting


logger = md_utils.util_logger(__name__)



# TODO Base Base class
class MdescBase(BaseEstimator,
                TransformerMixin):

    def __init__(self, modelobj):
        pass

    def fit(self, X, y=None, **kwargs):
        pass

    def fit_transform(self, X, y=None, **kwargs):
        """
        fit the model according to the given training data
        :param X:
        :param y:
        :param kwargs:
        :return:
        """
        pass

    def _base_runner(self,
                     df,
                     col,
                     groupby_var,
                     **kwargs):
        """
        Push grouping and iteration construct to core pandas. Normalize each column
        (leave categories as is and convert numeric cols to percentile bins). Apply
        transform function, and format final outputs.

        :param df: pd.DataFrame
        :param col: str - column currently being operated on
        :param groupby_var: str - groupby level
        :return: tuple of transformed data (return type, return data)
        :rtype: tuple
        """
        if col != groupby_var:
            res = (df.groupby(groupby_var)
                   .apply(MdescBase.revalue_numeric,
                          col)
                   .reset_index(drop=True)
                   .groupby([groupby_var, col])
                   .apply(self._transform_func,
                          groupby_var=groupby_var,
                          col=col,
                          output_df=kwargs.get('output_df', False))
                   .reset_index(drop=True)
                   .fillna('nan')
                   .round(self.round_num))
            out = ('res', res)

        else:
            logging.info(
                """Creating accuracy metric for 
                groupby variable: {}""".format(groupby_var))
            # create error metrics for slices of groupby data
            acc = md_utils.create_accuracy(self.model_type,
                                           self._cat_df,
                                           self.error_type,
                                           groupby=groupby_var)
            # append to insights dataframe placeholder
            out = ('insights', acc)

        return out


# TODO base class regressor

# TODO base class classification





class MdescBase(six.with_metaclass(ABCMeta,
                                   percentiles.Percentiles,
                                   BaseEstimator)):

    @abstractmethod
    def __init__(
            self,
            modelobj,
            model_df,
            ydepend,
            **kwargs):
        """
        MdescBase base class instantiation and parameter checking

        :param modelobj: fitted sklearn model object
        :param model_df: dataframe used for training sklearn object
        :param ydepend: str dependent variable
        :param cat_df: dataframe formatted with categorical dtypes specified,
                       and non-dummy categories
        :param keepfeaturelist: list of features to keep in output
        :param groupbyvars: list of groupby variables
        :param error_type: str aggregate error metric i.e. MSE, MAE, RMSE, MED, MEAN
                MSE - Mean Squared Error
                MAE - Mean Absolute Error
                RMSE - Root Mean Squared Error
                MED - Median Error
                MEAN - Mean Error
        :param autoformat_types: boolean to auto format categorical columns to objects
        :param round_num: round numeric columns to specified level for output
        :param verbose: set verbose level -- 0 = debug, 1 = warning, 2 = error
        """
        logger.setLevel(md_utils.Settings.verbose2log[verbose])
        logger.info('Initilizing {} class parameters'.format(self.__class__.__name__))

        super(MdescBase, self).__init__(
                                        cat_df,
                                        groupbyvars,
                                        round_num=round_num)

        self._model_df = model_df.copy(deep=True).reset_index(drop=True)
        self._keepfeaturelist = keepfeaturelist
        self._cat_df = cat_df
        self._modelobj = modelobj
        self.aggregate_func = aggregate_func
        self.error_type = error_type
        self.ydepend = ydepend
        self.groupbyvars = groupbyvars
        self.called_class = self.__class__.__name__
        self.agg_df = pd.DataFrame()
        self.raw_df = pd.DataFrame()
        self.round_num = round_num
        if autoformat_types is True:
            self._cat_df = formatting.autoformat_types(self._cat_df)

    def _validate_params(self):
        """
        private function to valide class attributes
        :return: NA
        """
        # check featurelist
        self._keepfeaturelist = checks.CheckInputs.check_keepfeaturelist(self._keepfeaturelist,
                                                                         self._cat_df)

        self._cat_df = checks.CheckInputs.check_cat_df(self._cat_df,
                                                       self._model_df)
        # check for classification or regression
        self._cat_df = formatting.subset_input(self._cat_df,
                                               self._keepfeaturelist,
                                               self.ydepend)

        self.predict_engine, self.model_type = checks.CheckInputs.is_regression(self._modelobj)

        # check groupby vars
        if not self.groupbyvars:
            # TODO add all data hack
            raise md_utils.ErrorWarningMsgs.error_msgs['groupbyvars']

        checks.CheckInputs.check_modelobj(self._modelobj)

        # get population percentiles
        self.population_percentiles()


    @staticmethod
    def revalue_numeric(data,
                        col):
        """
        revalue numeric columns with max value of percnetile group

        :param data: slice of data
        :param col: col value to revalue
        :return: revalued dataframe on column
        :rtype: pd.DataFrame
        """
        # ensure numeric dtype and slice is greater than 100 observations
        # otherwise return data as is
        if is_numeric_dtype(data.loc[:, col]) and data.shape[0] > 100:
            data['bins'] = pd.qcut(data[col],
                                   q=100,
                                   duplicates='drop',
                                   labels=False)
            # get max vals per group
            maxvals = (data.groupby('bins')[col]
                       .max()
                       .reset_index(name='maxcol'))
            # merge back
            data = (data.join(maxvals, on='bins', how='inner',
                              lsuffix='_left', rsuffix='_right')
                    .rename(columns={'bins_left': 'bins'}))
            # drop and rename columns
            data = (data
                    .drop(['bins', col], axis=1)
                    .rename(columns={'maxcol': col}))

        return data

    def _create_preds(self,
                      df):
        """
        create predictions based on model type. If classification, take corresponding
        probabilities related to the 1st class. If regression, take predictions as is.

        :param df: input dataframe
        :return: predictions
        :rtype: list
        """
        if self.model_type == 'classification':
            preds = self.predict_engine(df)[:, 1]
        else:
            preds = self.predict_engine(df)

        return preds

    def _plc_hldr_out(self,
                      insights_list,
                      results,
                      html_type):
        """
        format output df into nested json for html out

        :param insights_list: list of dataframes containing accuracy metrics
        :param results: list of results per iteration
        :param html_type:
        :return:
        """
        final_output = []

        aligned = formatting.FmtJson.align_out(results,
                                               html_type=html_type)
        # for each json element, flatten and append to final output
        for json_out in aligned:
            final_output.append(json_out)

        logging.info('Converting accuracy outputs to json format')
        # finally convert insights_df into json object
        # convert insights list to dataframe
        insights_df = pd.concat(insights_list)
        insights_json = formatting.FmtJson.to_json(insights_df.round(self.round_num),
                                                   html_type='accuracy',
                                                   vartype='Accuracy',
                                                   err_type=self.error_type,
                                                   ydepend=self.ydepend,
                                                   mod_type=self.model_type)
        # append to outputs
        final_output.append(insights_json)
        # append percentiles
        final_output.append(self.percentiles)
        # append groupby percentiles
        final_output.append(self.group_percentiles_out)
        # assign placeholder final outputs to class instance
        self.outputs = final_output







