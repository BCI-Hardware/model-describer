from functools import reduce

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from mdesc.utils import mdesc_exceptions
from mdesc.utils.mdesc_utils import load_wine
from mdesc.data.datavisualizer import DataVisualizer
from mdesc.utils.metrics import MetricMixin


def is_in_jupyter():
    """
    Check if user is running spaCy from a Jupyter notebook by detecting the
    IPython kernel. Mainly used for the displaCy visualizer.
    RETURNS (bool): True if in Jupyter, False if not.
    """
    try:
        from IPython import get_ipython
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
    except NameError:
        return False
    return False


class DataUtilities(object):

    @staticmethod
    def _revalue_numeric(input_arr):
        """
        convert input arr into percentile bins with percentile
        represented by max value of group
        """
        if is_numeric_dtype(input_arr) and input_arr.shape[0] > 100:
            percentile = np.nanpercentile(input_arr, interpolation='higher',
                                          q=list(range(100)))
            idx = np.searchsorted(percentile, input_arr, side='left',
                                  sorter=np.argsort(percentile))
            # remap values outside of range to max bin
            idx[idx == 100] = 99
            # return max value within percentile group
            input_arr = percentile[idx]

        return input_arr

    @staticmethod
    def create_percentiles(input_arr, percentile_list=None):
        if percentile_list is None:
            percentile_list = [0, 1, 10, 25, 50, 75, 90, 100]

        p_arr = np.nanpercentile(input_arr, q=percentile_list,
                                 axis=0, keepdims=False, interpolation='higher')

        str_p_list = ['{}%'.format(p) for p in percentile_list]
        return p_arr, percentile_list, str_p_list

    @staticmethod
    def create_continuous_percentiles(input_arr=None, feature_names=None,
                                      round_num=4):

        arr, percentile_list, str_p_list = DataUtilities.create_percentiles(input_arr)
        df = pd.DataFrame(arr, columns=feature_names)
        df['percentiles'] = str_p_list
        df = df.melt(id_vars='percentiles').round(decimals=round_num)
        z = df.groupby('variable').apply(lambda x: x.to_dict(orient='rows')).reset_index(name='perVar')
        p_out = {'Type': 'Percentile',
                 'Data': []}
        z['perVar'].apply(lambda x: p_out['Data'].extend(x))
        return (p_out, df)

    @staticmethod
    def create_continuous_groupby_percentiles(cur_arr, **kwargs):
        """

        :param cur_arr:
        :param kwargs:
        :return:

        Output Example
        -----------
        value percentile groupByVar           colname
        0.115         0%       high  volatile acidity
        0.160         1%       high  volatile acidity
        0.220        10%       high  volatile acidity
        """
        group_indices = kwargs.get('group_indices')
        group_level = kwargs.get('group_level')
        colname = kwargs.get('colname')
        cur_group_arr = cur_arr[group_indices]
        p_arr, percentile_list, str_p_list = DataUtilities.create_percentiles(cur_group_arr)
        df = pd.DataFrame(p_arr, columns=['value'])
        df['percentiles'] = str_p_list
        df['groupByVar'] = group_level
        df['colname'] = colname
        df = df.round(decimals=kwargs.get('round_num', 4))
        return df

    @staticmethod
    def isbinary(l1):
        """check if input list contains binary elements only"""
        return np.array_equal(l1, l1.astype(bool))

    @staticmethod
    def _reduce_multi_index(*args):
        """reduce list of slice indices to common elements"""
        return reduce(np.intersect1d, args)

    @staticmethod
    def _create_percentile_out(input_arr, feature_name,
                               percentiles=None):

        if percentiles is None:
            percentiles = [0, 1, 10, 25, 50, 75, 90, 100]

        all_percentiles = []
        for p in percentiles:
            percentile = '{}%'.format(p)
            percentile_value = np.percentile(input_arr, p)
            out = {'percentiles': percentile, 'value': percentile_value}
            all_percentiles.append(out)

        to_return = {'variable': feature_name,
                     'percentileList': all_percentiles}

        return to_return


class DataManager(DataVisualizer, MetricMixin):

    __datatypes__ = (pd.DataFrame, pd.Series, np.ndarray)

    def __init__(self, X=None, y=None, groupby_df=None,
                 target_name=None, target_classes=None,
                 feature_names=None, groupby_names=None,
                 model_type='regression', round_num=4,
                 class_type='sensitivity'):

        """
        Examples
        ---------
        >>> from mdesc.data.datamanager import DataManager
        >>> from mdesc.utils.mdesc_utils import load_wine
        >>> # featuredict - cat and continuous variables
        >>> wine = load_wine
        >>> groupby_df = wine.loc[:, ['Type', 'alcohol']]
        >>> X = wine.loc[:, wine.columns != 'quality']
        >>> y = wine.loc[:, 'quality'].values
        >>> X = pd.get_dummies(X)
        >>> DM = DataManager(X=X, y=y, groupby_df=groupby_df,
        >>>                 target_name='quality', groupby_names=None)
        >>> sub_groups = DM.create_sub_groups()
        >>> # iterate over sub groups and extract out key info
        >>> for group in sub_groups:
        >>>     group_level = group[0]
        >>>     group_col = group[1]
        >>>     colname = group[2]
        >>>     percentile_value = group[3]
        >>>     percentile_indices = group[4]

        :param X:
        :param y:
        :param groupby_df:
        :param target_name:
        :param target_classes:
        :param feature_names:
        :param groupby_names:
        :param model_type:
        :param round_num:
        :param class_type:
        """

        self.model_type = model_type
        self.feature_names = self._format_labels(feature_names)
        self.target_classes = self._check_target_classes(target_name, y)
        self.target_name = self._format_labels(target_classes)
        self.groupby_names = self._format_labels(groupby_names)
        self._X = self._check_X(X)
        self._y = self._check_y(y, X)
        self._groupby_df = self._check_groupby_df(groupby_df, X)
        self.round_num = round_num
        self.class_type = class_type
        self.continuous_indices = set()
        self._results = pd.DataFrame()
        self.accuracy = pd.DataFrame()
        self.p_group_df = pd.DataFrame()

        super(DataManager, self).__init__(round_num=round_num)

    def _format_labels(self, labels):
        if isinstance(labels, str) or labels is None:
            labels = [labels]
        return labels

    def _check_target_classes(self, target_classes, y):
        if self.model_type == 'classification':
            if target_classes is None:
                if all([isinstance(yval, str) for yval in np.unique(y)]):
                    if np.unique(y).shape[0] > 2:
                        raise ValueError("""Currently only supports binary classification problems.
                                        \nnum_classes: {}""".format(np.unique(y).shape[0]))
                    target_classes = {val: idx for idx, val in enumerate(np.unique(y))}
                else:
                    target_classes = {val: val for val in enumerate(np.unique(y))}
            return target_classes

    def _check_groupby_df(self, groupby_df, X):
        if groupby_df is None:
            groupby_df = np.array([1] * len(X)).reshape(-1, 1)
            self.groupby_names = ['all_values']

        if groupby_df.shape[0] != X.shape[0]:
            err_msg = """groupby_df shape not aligned with X.
                        Input X: {},
                        Input groupby_df: {}""".format(len(X), len(groupby_df))

            raise mdesc_exceptions.InputGroupbyDfError(err_msg)

        if isinstance(groupby_df, pd.DataFrame) and self.groupby_names[0] is None:
            self.groupby_names = groupby_df.columns
            groupby_df = groupby_df.values

        if isinstance(groupby_df, pd.DataFrame):
            groupby_df = groupby_df.values

        if self.groupby_names[0] is None:
            self.groupby_names = ['group_{}'.format(idx) for idx, _ in enumerate(list(range(groupby_df.shape[1])))]

        if self.groupby_names[0] is not None:
            if len(self.groupby_names) != groupby_df.shape[1]:
                err_msg = """groupby_names does not have the same number of labels as columns in groupby_df.
                            groupby_names: {}; groupby_df: {}""".format(len(self.groupby_names), groupby_df.shape[1])
                raise mdesc_exceptions.LabelShapeError(err_msg)

        if groupby_df.ndim == 1:
            groupby_df = groupby_df.reshape(-1, 1)
        return groupby_df

    def _check_X(self, X):
        if not isinstance(X, self.__datatypes__):
            err_msg = 'Invalid Data: expected data to be a numpy array or pandas dataframe but got ' \
                      '{}'.format(type(X))
            raise (mdesc_exceptions.InputXError(err_msg))
        ndim = len(X.shape)
        #self.logger.debug("__init__ data.shape: {}".format(X.shape))

        if isinstance(X, pd.DataFrame) and self.feature_names[0] is None:
            self.feature_names = X.columns.tolist()

        if self.feature_names[0] is None:
            self.feature_names = ['feature_{}'.format(idx) for idx, val in enumerate(list(range(X.shape[1])))]

        self.feature_names = np.array(self.feature_names)

        if self.feature_names.shape[0] != X.shape[1]:
            err_msg = """feature_names not the same shape as columns in X
                        feature_names: {}, X: {}""".format(self.feature_names.shape[0],
                                                           X.shape[1])
            raise (mdesc_exceptions.FeatureNamesShapeError(err_msg))


        if isinstance(X, pd.DataFrame):
            X = X.values

        if ndim == 1:
            X = X[:, np.newaxis]

        elif ndim >= 3:
            err_msg = "Invalid Data: expected data to be 1 or 2 dimensions, " \
                      "Data.shape: {}".format(ndim)
            raise (mdesc_exceptions.InputXError(err_msg))
        return X

    def _check_y(self, y, X):
        """
        convert y to ndarray
        If y is a dataframe:
            return df.values as ndarray
        if y is a series
            return series.values as ndarray
        if y is ndarray:
            return self
        if y is a list:
            return as ndarray
        :param y:
        :param X:
        :return:
        """
        if y is None:
            return y

        if len(X) != len(y):
            raise mdesc_exceptions.InputXYError(
                """
                Input X and y shapes are misaligned. Input X: {},
                Input Y: {}
                """.format(X.shape[0], y.shape[0])
            )
        if isinstance(y, (pd.DataFrame, pd.Series)):
            if self.target_name[0] is None:
                target_name = getattr(y, 'columns', y.name)
                if not isinstance(target_name, str):
                    if target_name is None:
                        target_name = 'target'
                    else:
                        target_name = target_name[0]
                self.target_name = target_name
            return y.values
        elif isinstance(y, np.ndarray):
            return y
        elif isinstance(y, list):
            return np.array(y)
        else:
            raise ValueError("Unrecognized type for y: {}".format(type(y)))

    def return_group_array(self):
        """iterate over groupby dataframe and levels within groups"""
        for idx, group_col in enumerate(self.groupby_names):
            print(idx, group_col)
            group_arr = self._groupby_df[:, idx]
            for group_level in np.unique(group_arr):
                print(group_level)
                group_indices = np.where(group_arr == group_level)[0] # unpack tuple
                output_dict = {}
                output_dict['group_arr'] = group_arr
                output_dict['group_level'] = group_level
                output_dict['groupby_var'] = group_col
                output_dict['group_indices'] = group_indices
                yield output_dict

    def yield_continuous_percentile(self, full_input_arr=None,
                                     row_indices=None):
        """
        construct percentile groups from continuous variable

        :param full_input_arr: continuous input array (n_rows)
        :param row_indices: subsetter indices (i.e. groupby level)
        :return: tuple (indices of rows corresponding to particular percentile,
            percentile value)
        """

        full_input_arr = np.copy(full_input_arr)
        if row_indices is None:
            row_indices = np.nonzero(full_input_arr[:, 0])
        full_input_arr[row_indices] = DataUtilities._revalue_numeric(full_input_arr[row_indices])

        for percentile in np.unique(full_input_arr[row_indices]):
            percentile_indices = np.where(full_input_arr[row_indices].ravel()==percentile)[0]
            # get the indices that align with percentile from original indices
            percentile_indices = row_indices[percentile_indices]
            yield (percentile_indices, percentile)

    def create_group_accuracy(self, group_err_arr=None,
                              **kwargs):

        errors = kwargs.get('errors')
        group_err_arr = errors[kwargs.get('group_indices')]
        error_metric = self.metric_all(group_err_arr,
                                       metric=kwargs.get('error_type'))

        row = pd.DataFrame({'Yvar': self.target_name,
                            'ErrType': kwargs.get('error_type'),
                            'Type': 'Accuracy',
                            'groupByValue': kwargs.get('group_level'),
                            kwargs.get('error_type'): error_metric,
                           'Total': group_err_arr.shape[0],
                           'groupByVarName': kwargs.get('groupby_var')}, index=[0])

        self.accuracy = self.accuracy.append(row)

    def create_sub_groups(self, **kwargs):
        """
        create slices of full data based on grouping levels and percentiles of continuous values

        :return: tuple - (group level, grouping column, continuous column name,
            percentile value, indices of rows corresponding to group level and percentile)
        """
        for group_container in self.return_group_array():
            # group_arr, group_level, group_col, group_indices = group_container

            # create group accuracy
            if kwargs.get('errors', None) is not None:
                group_container['errors'] = kwargs.get('errors')
                group_container['error_type'] = kwargs.get('error_type')
                group_container['round_num'] = self.round_num
                self.create_group_accuracy(**group_container)
                group_container.pop('errors')

            for idx, colname in enumerate(self.feature_names):
                cur_arr = self._X[:, idx]
                group_container['colname'] = colname
                # create group level percentiles

                if DataUtilities.isbinary(cur_arr):
                    # TODO add future support for categorical variables
                    continue
                group_container['dtype'] = 'Continuous'
                # construct percentile values within column within groupby level
                percentile_groupby = DataUtilities.create_continuous_groupby_percentiles(cur_arr,
                                                                                         **group_container)
                self.p_group_df = self.p_group_df.append(percentile_groupby)

                self.continuous_indices.add(idx)

                percentile_generator = self.yield_continuous_percentile(full_input_arr=cur_arr,
                                                                        row_indices=group_container['group_indices'])

                for percentile_indices, percentile_value in percentile_generator:
                    group_container['percentile_indices'] = percentile_indices
                    group_container['percentile_value'] = percentile_value
                    #yield (group_level, group_col, colname, percentile_value, percentile_indices)
                    yield group_container

        # create top level percentiles
        p_input_fnames = self.feature_names[list(self.continuous_indices)]
        p_input_arr = self.X[:, list(self.continuous_indices)]
        self.cont_percentile_json, self.cont_percentile_df = DataUtilities.create_continuous_percentiles(input_arr=p_input_arr,
                                                                                                         feature_names=p_input_fnames,
                                                                                                         round_num=self.round_num)

    @property
    def groupby_df(self):
        return self._groupby_df

    @groupby_df.setter
    def groupby_df(self, value):
        groupby_df = self._check_groupby_df(value, self._X)
        self._groupby_df = groupby_df

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("""results must be pd.DataFrame. Got type: {}""".format(type(value)))
        self._results = value.round(decimals=self.round_num)
