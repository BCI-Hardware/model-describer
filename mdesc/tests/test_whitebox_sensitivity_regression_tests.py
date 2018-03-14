#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from mdesc.utils import utils as wb_utils
from mdesc.eval import SensitivityViz

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

class TestWBBaseMethods(unittest.TestCase):

    def setUp(self):
        # create wine dataset
        try:
            wine = pd.read_csv('testdata/wine.csv')
        except FileNotFoundError:
            wine = pd.read_csv('/home/travis/build/DataScienceSquad/model-describer/mdesc/tests/testdata/wine.csv')

        # init randomforestregressor
        modelObjc = RandomForestRegressor()

        ydepend = 'quality'
        # create second categorical variable by binning
        wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
        # subset dataframe down
        wine_sub = wine.copy(deep=True)

        mod_df = pd.get_dummies(wine_sub.loc[:, wine_sub.columns != ydepend])

        modelObjc.fit(mod_df,
                      wine_sub.loc[:, ydepend])

        keepfeaturelist = ['fixed acidity',
                           'Type',
                           'quality',
                           'volatile.acidity.bin',
                           'alcohol',
                           'density',
                           'sulphates']

        wine_sub['alcohol'] = wine_sub['alcohol'].astype('object')

        self.WB = SensitivityViz(modelobj=modelObjc,
                                 model_df=mod_df,
                                 ydepend=ydepend,
                                 cat_df=wine_sub,
                                 groupbyvars=['Type'],
                                 keepfeaturelist=keepfeaturelist,
                                 verbose=None,
                                 autoformat_types=True)

        self.wine = wine
        self.mod_df = mod_df

    def test_transform_func_predictedYSmooth(self):
        """test predictedYSmooth present after _transform_func called"""
        self.WB._validate_params()

        group = self.wine.groupby('Type').get_group('White')

        group['diff'] = np.random.uniform(-1, 1, group.shape[0])

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        col = 'density'

        res = self.WB._transform_func(group,
                                          col=col,
                                          groupby_var='Type')

        self.assertIn('predictedYSmooth',
                      res.columns.tolist(),
                      """predictedYSmooth not present in output df after
                      _transform_func called""")

    def test_transform_func_predictedYSmooth_val(self):
        """test median predictedYSmooth is returned after transform_function called"""
        self.WB._validate_params()
        group = self.wine.groupby('Type').get_group('White')

        group['diff'] = np.random.uniform(-1, 1, group.shape[0])

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        correct = np.nanmedian(group['diff'])

        col = 'density'

        res = self.WB._transform_func(group,
                                          col=col,
                                          groupby_var='Type')

        self.assertEqual(res['predictedYSmooth'].values.tolist()[0],
                         correct,
                         """unexpected value for predictedYSmooth - should be median default""")

    def test_predict_synthetic_inc_val(self):
        """test incremental val for predict_synthetic"""

        # assign preds
        self.WB._validate_params()
        self.WB._cat_df['predictedYSmooth'] = np.random.rand(self.WB._cat_df.shape[0])
        incremental_val, diff, mask = self.WB._predict_synthetic(col='Type')

        self.assertEqual(incremental_val,
                         self.wine['Type'].mode().tolist()[0],
                         """incorrect categorical incrementing value detected""")

    def test_predict_synthetic_inc_val_continuous(self):
        """test incremental val for predict_synthetic continuous variable"""

        # assign preds
        self.WB._validate_params()
        self.WB._cat_df['predictedYSmooth'] = np.random.rand(self.WB._cat_df.shape[0])
        incremental_val, diff, mask = self.WB._predict_synthetic(col='sulphates')

        self.assertEqual(incremental_val,
                         self.wine['sulphates'].std(),
                         """incorrect continuous incrementing value detected""")

