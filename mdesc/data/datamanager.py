from functools import reduce

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from mdesc.utils import mdesc_exceptions


def _revalue_numeric(input_arr):
    """
    convert input arr into percentile bins with percentile
    represented by max value of group
    """
    if is_numeric_dtype(input_arr) and input_arr.shape[0] > 100:
        percentile = np.percentile(input_arr, interpolation='higher',
                                   q=list(range(100)))
        idx = np.searchsorted(percentile, input_arr, side='left',
                              sorter=np.argsort(percentile))
        # remap values outside of range to max bin
        idx[idx == 100] = 99
        # return max value within percentile group
        input_arr = percentile[idx]

    return input_arr

def is_in_jupyter():
    """Check if user is running spaCy from a Jupyter notebook by detecting the
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

    def _revalue_numeric(self, input_arr):
        """
        convert input arr into percentile bins with percentile
        represented by max value of group
        """
        if is_numeric_dtype(input_arr) and input_arr.shape[0] > 100:
            percentile = np.percentile(input_arr, interpolation='higher',
                                       q=list(range(100)))
            idx = np.searchsorted(percentile, input_arr, side='left',
                                  sorter=np.argsort(percentile))
            # remap values outside of range to max bin
            idx[idx == 100] = 99
            # return max value within percentile group
            input_arr = percentile[idx]

        return input_arr

    def isbinary(self, l1):
        """check if input list contains binary elements only"""
        return np.array_equal(l1, l1.astype(bool))

    def _reduce_multi_index(self, *args):
        """reduce list of slice indices to common elements"""
        return reduce(np.intersect1d, args)


class DataManager(DataUtilities):

    __datatypes__ = (pd.DataFrame, pd.Series, np.ndarray)

    def __init__(self, X=None, y=None, groupby_df=None,
                 target_name=None, target_classes=None,
                 feature_names=None, groupby_names=None,
                 model_type='regression'):

        self.model_type = model_type
        self.feature_names = self._format_labels(feature_names)
        self.target_classes = self._check_target_classes(target_name, y)
        self.target_name = self._format_labels(target_classes)
        self.groupby_names = self._format_labels(groupby_names)

        self._X = self._check_X(X)
        self._y = self._check_y(y, X)
        self._groupby_df = self._check_groupby_df(groupby_df, X)

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
            print(groupby_df.shape)
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
                yield (group_arr, group_level, group_col, group_indices)

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
        full_input_arr[row_indices] = _revalue_numeric(full_input_arr[row_indices])

        for percentile in np.unique(full_input_arr[row_indices]):
            percentile_indices = np.where(full_input_arr[row_indices].ravel()==percentile)[0]
            # get the indices that align with percentile from original indices
            percentile_indices = row_indices[percentile_indices]
            yield (percentile_indices, percentile)


    def create_sub_groups(self):
        """
        create slices of full data based on grouping levels and percentiles of continuous values

        :return: tuple - (group level, grouping column, continuous column name,
            percentile value, indices of rows corresponding to group level and percentile)
        """
        for group_container in self.return_group_array():
            group_arr, group_level, group_col, group_indices = group_container

            for idx, colname in enumerate(self.feature_names):
                cur_arr = self._X[:, idx]
                if self.isbinary(self._X[:, idx]):
                    # TODO add future support for categorical variables
                    continue

                percentile_generator = self.yield_continuous_percentile(full_input_arr=cur_arr,
                                                                        row_indices=group_indices)

                for percentile_indices, percentile_value in percentile_generator:
                    yield (group_level, group_col, colname, percentile_value, percentile_indices)

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



"""
# featuredict - cat and continuous variables
wine = pd.read_csv('debug/wine.csv')

groupby_df = wine.loc[:, ['Type', 'alcohol']]

X = wine.loc[:, wine.columns != 'quality']
y = wine.loc[:, 'quality'].values

X = pd.get_dummies(X)

DM = DataManager(X=X, y=y, groupby_df=groupby_df,
                 target_name='quality', groupby_names=None)

sub_groups = DM.create_sub_groups()
group_level, group_col, colname, percentile_value, percentile_indices = next(sub_groups)
percentile_indices

wine.iloc[percentile_indices]

DM.y = wine.loc[:, 'Type'].values

percentile_value
wine.iloc[percentile_indices][['Type', 'volatile acidity']]
"""

