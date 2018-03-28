from pandas.api.types import is_object_dtype, is_numeric_dtype
import pandas as pd
import numpy as np
import itertools
import logging

from mdesc.base import MdescBase
from mdesc.utils.fmt_model_outputs import fmt_sklearn_preds
from mdesc.utils import utils as md_utils
from mdesc.utils.categorical_conversions import pandas_switch_modal_dummy

logger = md_utils.util_logger(__name__)


class ErrorViz(MdescBase):

    def __init__(
                    self,
                    modelobj,
                    model_df,
                    ydepend,
                    cat_df=None,
                    groupbyvars=None,
                    **kwargs):

        """
        :param modelobj: sklearn model object
        :param model_df: Pandas Dataframe used to build/train model
        :param ydepend: Y dependent variable
        :param cat_df: Pandas Dataframe of raw data - with categorical datatypes
        :param keepfeaturelist: Subset and rename columns
        :param groupbyvars: grouping variables
        :param aggregate_func: numpy aggregate function like np.mean
        :param dominate_class: in the case of binary classification, class of interest
            to measure probabilities from
        :param round_num: round numeric values for output
        :param autoformat: experimental autoformatting of dataframe
        :param verbose: Logging level
        """
        super(ErrorViz, self).__init__(
                                            modelobj,
                                            model_df,
                                            ydepend,
                                            cat_df=cat_df,
                                            groupbyvars=groupbyvars,
                                            error_type=kwargs.get('error_type', 'MSE'),
                                            keepfeaturelist = kwargs.get('keepfeaturelist', None),
                                            aggregate_func=kwargs.get('aggregate_func', np.nanmedian),
                                            autoformat_types=kwargs.get('autoformat_types', True),
                                            round_num=kwargs.get('round_num', 4),
                                            verbose=kwargs.get('verbose', None))

    def _transform_func(self,
                        group,
                        groupby_var='Type',
                        col=None,
                        output_df=False):
        """
        aggregate and format measures within slices of data for final output

        :param group: current slice of dataframe
        :param groupby_var: str - groupby variable
        :param col: str - column being operated on
        :param output_df: bool - build and track raw_df and agg_df
        :return: formatted and aggregated final output dataframe
        :rtype: pd.DataFrame
        """
        error_arr = group['errors'].values
        # subtract errors from group median
        if self.model_type == 'classification':
            # get user defined aggregate (central values) of the errors
            agg_errors = np.nanmedian(error_arr)
            # subtract the aggregate value for the group from the errors
            error_arr = agg_errors - error_arr

        # aggregate errors
        agg_errors = pd.DataFrame({col: group[col].mode(),
                                   'groupByValue': group[groupby_var].mode(),
                                   'groupByVarName': groupby_var,
                                   'predictedYSmooth': np.nanmedian(group['predictedYSmooth']),
                                   'errPos': np.nanmedian(error_arr[error_arr >= 0]),
                                   'errNeg': np.nanmedian(error_arr[error_arr <= 0])}, index=[0])

        # fmt and append agg_df to instance attribute
        # agg_df
        if output_df:
            self._fmt_agg_df(col=col,
                             agg_errors=agg_errors)

        return agg_errors

    def run(self,
            output_type='html',
            **kwargs):
        """
        main run engine. Iterate over columns specified in keepfeaturelist,
        and perform anlaysis
        :param output_type: str output type:
                html - save html to output_path
                raw_data - return raw analysis dataframe
                agg_data - return aggregate analysis dataframe
        :param output_path: - fpath to save output
        :return: pd.DataFrame or saved html output
        :rtype: pd.DataFrame or .html
        """
        self._validate_params()
        # run the prediction function first to assign the errors to the dataframe
        self._cat_df = fmt_sklearn_preds(self.predict_engine,
                                         self._modelobj,
                                         self._model_df,
                                         self._cat_df,
                                         self.ydepend,
                                         self.model_type)
        # placeholders
        placeholder = {'res': [],
                       'insights': []}

        logging.info("""Running main program. Iterating over 
                            columns and applying functions depednent on datatype""")

        not_in_cols = ['errors', 'predictedYSmooth', self.ydepend]

        # filter columns to iterate through
        to_iter_cols = self._cat_df.columns[~self._cat_df.columns.isin(not_in_cols)]

        # create combinations of groupby and columns
        all_iter = list(itertools.product(to_iter_cols, self.groupbyvars))

        pbar = md_utils.progress_bar()
        progress_bar = pbar(total=len(all_iter))

        for (col, groupby_var) in itertools.product(to_iter_cols, self.groupbyvars):

            col_indices = [col, 'errors', 'predictedYSmooth', groupby_var]

            key, value = self._base_runner(self._cat_df.loc[:, col_indices],
                                           col,
                                           groupby_var)

            placeholder[key].append(value)

            #logger.info("""Run processed - Col: {} - groupby_var: {}""".format(col, groupby_var))

            progress_bar.update(1)

        # convert placeholders to final output
        self._plc_hldr_out(placeholder['insights'],
                           placeholder['res'],
                           html_type='error')

        # save outputs if specified
        outputs = self._save(output_type=output_type,
                             fpath=kwargs.get('output_path', None))
        # underlying data select, return it
        if isinstance(outputs, pd.DataFrame):
            return outputs

class SensitivityViz(MdescBase):

    def __init__(self,
                 modelobj,
                 model_df,
                 ydepend,
                 cat_df=None,
                 groupbyvars=None,
                 **kwargs):
        """
        :param modelobj: sklearn model object
        :param model_df: Pandas Dataframe used to build/train model
        :param ydepend: Y dependent variable
        :param cat_df: Pandas Dataframe of raw data - with categorical datatypes
        :param featuredict: Subset and rename columns
        :param groupbyvars: grouping variables
        :param aggregate_func: function to aggregate sensitivity results by group
        :param autoformat: experimental auto formatting of dataframe
        :param round_num: round numeric values for output
        :param verbose: Logging level
        :param std_num: Standard deviation adjustment
        """
        logger.setLevel(md_utils.Settings.verbose2log[kwargs.get('verbose', None)])
        logger.info('Initilizing {} parameters'.format(self.__class__.__name__))

        self.std_num = kwargs.get('std_num', 1)

        super(SensitivityViz, self).__init__(
                                                    modelobj,
                                                    model_df,
                                                    ydepend,
                                                    cat_df=cat_df,
                                                    groupbyvars=groupbyvars,
                                                    keepfeaturelist=kwargs.get('keepfeaturelist', None),
                                                    aggregate_func=kwargs.get('aggregate_func', np.nanmedian),
                                                    error_type=kwargs.get('error_type', 'MEAN'),
                                                    autoformat_types=kwargs.get('autoformat_types', True),
                                                    round_num=kwargs.get('round_num', 4),
                                                    verbose=kwargs.get('verbose', None))

    def _transform_func(self,
                        group,
                        groupby_var='Type',
                        col=None,
                        output_df=False):
        """
        aggregate and format measures within slices of data for final output

        :param group: current slice of dataframe
        :param groupby_var: str - groupby variable
        :param col: str - column being operated on
        :param output_df: bool - build and track raw_df and agg_df
        :return: formatted and aggregated final output dataframe
        :rtype: pd.DataFrame
        """
        assert 'errors' in group.columns, 'errors needs to be present in dataframe slice'
        logger.info("""Processing -- groupby_var: {} -- col: {} -- group shape: {}""".format(groupby_var,
                                                                                                            col,
                                                                                                            group.shape))

        # append raw_df to instance attribute raw_df
        if output_df:
            self._fmt_raw_df(col=col,
                             groupby_var=groupby_var,
                             cur_group=group)
        # aggregate errors
        agg_errors = pd.DataFrame({col: group[col].mode(),
                                   'groupByValue': group[groupby_var].mode(),
                                   'groupByVarName': groupby_var,
                                   'predictedYSmooth': np.nanmedian(group['diff'])}, index=[0])

        # fmt and append agg_df to instance
        # agg_df
        if output_df:
            self._fmt_agg_df(col=col,
                             agg_errors=agg_errors)

        return agg_errors

    def run(self,
            output_type='html',
            progbar=True,
            **kwargs):
        """
        main run engine. Iterate over columns specified in keepfeaturelist,
        and perform anlaysis
        :param output_type: str output type:
                html - save html to output_path
                raw_data - return raw analysis dataframe
                agg_data - return aggregate analysis dataframe
        :param output_path: - fpath to save output
        :return: pd.DataFrame or saved html output
        :rtype: pd.DataFrame or .html
        """
        # if output_type is a data format, force output_df to True and throw warning
        if output_type in ['raw_data', 'agg_data'] and kwargs.get('output_df') == False:
            kwargs['output_df'] = True
            raise Warning("""output_df must be set to True when returning dataframe. Forcing 
                            to true""")


        self._validate_params()
        # run the prediction function first to assign the errors to the dataframe
        self._cat_df = fmt_sklearn_preds(self.predict_engine,
                                         self._modelobj,
                                         self._model_df,
                                         self._cat_df,
                                         self.ydepend,
                                         self.model_type)
        # placeholders
        placeholder = {'res': [],
                       'insights': []}

        logging.info("""Running main program. Iterating over 
                            columns and applying functions depednent on datatype""")

        # filter cols to iterate over
        to_iter = [val for val in self._keepfeaturelist if val != self.ydepend]
        # create col, groupby combos
        all_iter = list(itertools.product(to_iter, self.groupbyvars))
        # create container with synthetic prediction difference, row mask,
        # and incremental val
        preds_container = self._preds_container(to_iter)
        # import pbar
        if progbar:
            pbar = md_utils.progress_bar()
            progress_bar = pbar(total=len(all_iter))

        for idx, (col, groupby_var) in enumerate(all_iter, 1):
            col_indices = [col, 'errors', 'predictedYSmooth', groupby_var, 'diff']

            # pull incremental val, diff, and mask from container
            incremental_val, diff, mask = preds_container[col]

            # update differences
            self._cat_df['diff'] = diff

            key, value = self._base_runner(self._cat_df.loc[mask, col_indices],
                                           col,
                                           groupby_var,
                                           **kwargs)
            # assign incremental value for output formatting
            if key == 'res':
                value['incremental_val'] = incremental_val

            placeholder[key].append(value)

            #logger.info("""Run processed - Col: {} - groupby_var: {}""".format(col, groupby_var))
            if progbar:
                progress_bar.update(1)

        # convert placeholders to final output
        self._plc_hldr_out(placeholder['insights'],
                           placeholder['res'],
                           html_type='sensitivity')

        # save outputs if specified
        self._save(output_type=output_type,
                   fpath=kwargs.get('output_path', None))

    def _preds_container(self,
                         to_iter):
        """
        create lookup dict for each col in keepfeaturelist which contains
        synthetic predictions difference from original preds

        :param to_iter: list of columns
        :return: dict lookup for each column
        :rtype: dict
        """
        container = {}
        for col in to_iter:
            incremental_val, diff, mask = self._predict_synthetic(col)

            container[col] = (incremental_val, diff, mask)

        return container

    def _predict_synthetic(self,
                           col):
        """
        In the continuous case, the standard deviation is determined by the values of
        the continuous column. This is multipled by the user defined std_num and applied
        to the values in the continuous column. New predictions are generated on this synthetic
        dataset, and the difference between the original predictions and the new predictions are
        captured and assigned.

        :param col: current column being operated on
        :param copydf: deep copy of cat_df
        :param col_indices: column indices to include
        :return: incremental bump value, sensitivity dataframe
        """
        copydf = self._model_df.copy(deep=True)
        if is_numeric_dtype(self._cat_df.loc[:, col]):
            incremental_val = copydf[col].std() * self.std_num
            # tweak the currently column by the incremental_val
            copydf[col] = copydf[col] + incremental_val
            mask = pd.Series([True] * copydf.shape[0])

        else:
            # switch modal column for predictions and subset
            # rows that are not already the mode value
            incremental_val, copydf, mask = pandas_switch_modal_dummy(col,
                                                                    self._cat_df,
                                                                    copydf)

        # make predictions with the switches to the dataset
        new_preds = self._create_preds(copydf)

        # calculate difference between actual predictions and new_predictions
        diff = new_preds - self._cat_df['predictedYSmooth']

        return incremental_val, diff, mask