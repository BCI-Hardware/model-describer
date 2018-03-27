# !/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
import logging
import six

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from mdesc.utils import utils as md_utils
from mdesc.utils import check_utils as checks
from mdesc.utils import percentiles
from mdesc.utils import formatting


logger = md_utils.util_logger(__name__)


class PrettyListMixn(object):
    """
    class mix-in for pretty printing of class attrs
    """

    def _getnames(self):
        """
        Retrieve class attributes

        :return: str - formatted class attributes
        :rtype: str
        """
        return ''.join(['\t{}={}\n'.format(attr, self.__dict__[attr])
                        for attr in sorted(self.__dict__)
                        if not isinstance(self.__dict__[attr], pd.DataFrame)])

    def __str__(self):
        """
        represent class attributes

        :return: str - formatted class attributes
        :rtype: str
        """
        return '<Instance of {}, address: {}\n{}>'.format(self.__class__.__name__,
                                                          id(self),
                                                          self._getnames())


class MdescBase(six.with_metaclass(ABCMeta,
                                   percentiles.Percentiles,
                                   PrettyListMixn)):

    @abstractmethod
    def __init__(
            self,
            modelobj,
            model_df,
            ydepend,
            cat_df=None,
            groupbyvars=None,
            keepfeaturelist=None,
            aggregate_func=np.nanmedian,
            error_type='RMSE',
            autoformat_types=False,
            round_num=4,
            verbose=None):
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
        logger.info('Initilizing {} class parmaetres'.format(self.__class__.__name__))

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

    @abstractmethod
    def _transform_func(self,
                        group,
                        groupby_var='Type',
                        col=None,
                        output_df=False):
        pass

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

    @abstractmethod
    def run(self,
            output_type='html',
            **kwargs):
        pass

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

    def _fmt_raw_df(self,
                    col,
                    groupby_var,
                    cur_group):
        """
        format raw_df by renaming various columns and appending to instance
        raw_df

        :param col: str current col being operated on
        :param groupby_var: str current groupby variable
        :param cur_group: current slice of group data
        :return: Formatted dataframe
        :rtype: pd.DataFrame
        """
        logger.info(
            """Formatting specifications - col: {} - groupbvy_var: {} - cur_group shape: {}""".format(col, groupby_var,
                                                                                                      cur_group.shape))

        # take copy
        group_copy = cur_group.copy(deep=True)
        # reformat current slice of data for raw_df
        raw_df = group_copy.rename(columns={col: 'col_value',
                                            groupby_var: 'groupby_level'}).reset_index(drop=True)

        raw_df['groupByVar'] = groupby_var
        raw_df['col_name'] = col
        if 'fixed_bins' in raw_df.columns:
            del raw_df['fixed_bins']

        # self.raw_df = self.raw_df.append(raw_df)
        self.raw_df = pd.concat([self.raw_df, raw_df])

    def _fmt_agg_df(self,
                    col,
                    agg_errors):
        """
        format agg_df by renaming various columns and appending to
        instance agg_df

        :param col: str current col being operated on
        :param agg_errors: aggregated error data for group
        :return: Formatted dataframe
        :rtype: pd.DataFrame
        """

        logger.info("""Formatting specifications - col: {} - agg_errors shape: {}""".format(col, agg_errors.shape))

        group_copy = agg_errors.copy(deep=True)
        debug_df = group_copy.rename(columns={col: 'col_value'})
        debug_df['col_name'] = col
        # self.agg_df = self.agg_df.append(debug_df)
        self.agg_df = pd.concat([self.agg_df, debug_df])

    def _save(self,
              output_type,
              **kwargs):
        """
        save html output to disc

        :param fpath: file path to save html file to
        """
        if output_type == 'html':
            logging.info("""creating html output for type: {}""".format(md_utils.Settings.html_type[self.called_class]))

            # tweak self.ydepend if classification case (add dominate class)
            if self.model_type == 'classification':
                ydepend_out = '{}: {}'.format(self.ydepend, self._modelobj.classes_[1])
            else:
                # regression case
                ydepend_out = self.ydepend

            # create HTML output
            html_out = formatting.HTML.fmt_html_out(
                str(self.outputs),
                ydepend_out,
                htmltype=md_utils.Settings.html_type[self.called_class])
            # save html_out to disk
            with open(kwargs.get('fpath', ValueError('Must specify fpath when saving as html')), 'w') as outfile:
                logging.info("""Writing html file out to disk""")
                outfile.write(html_out)
        if output_type == 'raw_data':
            return self.raw_df
        if output_type == 'agg_data':
            return self.agg_df

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




