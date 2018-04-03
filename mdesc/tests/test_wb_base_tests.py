#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from mdesc.base import MdescBase
from mdesc.eval import ErrorViz
from mdesc.utils import utils as wb_utils

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
                           'sulphates']

        wine_sub['alcohol'] = wine_sub['alcohol'].astype('object')

        self.wine = wine
        self.model_df = mod_df

        self.WB = ErrorViz(modelobj=modelObjc,
                           model_df=mod_df,
                           ydepend=ydepend,
                           cat_df=wine_sub,
                           groupbyvars=['Type'],
                           keepfeaturelist=keepfeaturelist,
                           verbose=None,
                           autoformat_types=True)

    def test_fmt_raw_df_groupByVar(self):
        """test fmt_raw_df inserts groupByVar column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_raw_df(col='alcohol',
                      groupby_var='Type',
                      cur_group=df)

        self.assertIn('groupByVar',
                      self.WB.raw_df.columns.tolist(),
                      """groupByVar not in WB.raw_df after fmt_raw_df run""")

    def test_fmt_agg_df_col_value(self):
        """test fmt_agg_df inserts col_value column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_agg_df(col='alcohol',
                           agg_errors=df)

        self.assertIn('col_value',
                      self.WB.agg_df.columns.tolist(),
                      """col_value not in WB.agg_df after fmt_agg_df run""")

    def test_fmt_agg_df_col_name(self):
        """test fmt_agg_df inserts col_name column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_agg_df(col='alcohol',
                           agg_errors=df)

        self.assertIn('col_name',
                      self.WB.agg_df.columns.tolist(),
                      """col_name not in WB.agg_df after fmt_agg_df run""")


    def test_base_run(self):
        """test that run assigns outputs when called"""

        self.WB.run(output_type='agg_data')

        self.assertTrue(hasattr(self.WB, 'outputs'),
                        """WB does not have attribute outputs after 
                        .run() called""")

    def test_base_run_output_type(self):
        """test output type after run called"""

        self.WB.run(output_type='raw_data')

        self.assertIsInstance(self.WB.outputs,
                              list,
                              """WB.outputs not of type list after .run() 
                              called""")

    def test_base_run_output_type_raw_df(self):
        """test output type after run called with output_type='raw_data'"""

        res = self.WB.run(output_type='raw_data')

        self.assertIsInstance(res,
                              pd.DataFrame,
                              """Returned type after .run() called not pd.DataFrame""")

    def test_base_run_output_type_agg_df(self):
        """test output type after run called with output_type='agg_data'"""

        res = self.WB.run(output_type='agg_data')

        self.assertIsInstance(res,
                              pd.DataFrame,
                              """Returned type after .run() called not pd.DataFrame""")

    def test_revalue_bins(self):
        """test binning from revalue function"""
        df = pd.DataFrame({'col1': list(range(1, 101)) * 10,
                           'col2': np.random.rand(1000)})

        res = self.WB.revalue_numeric(df,
                                      'col1')

        results = res.sort_values('col1')['col1'].unique().tolist()

        self.assertEqual(results,
                         list(range(1, 101)),
                         """revalue_numeric didn't return proper bins""")

    def test_preds(self):
        """test predictions from _create_preds"""
        self.WB._validate_params()
        preds = self.WB._modelobj.predict(self.model_df).tolist()
        preds2 = self.WB._create_preds(self.model_df).tolist()
        self.assertEqual(preds,
                         preds2,
                         """unequal predictions created""")

    def test_base_runner_res_out(self):
        """test base_runner output of results"""
        self.WB._validate_params()
        df = self.wine.copy(deep=True)
        df['errors'] = np.random.uniform(-1, 1, df.shape[0])
        df['predictedYSmooth'] = np.random.rand(df.shape[0])
        out = self.WB._base_runner(df,
                             'sulphates',
                             'Type')

        self.assertEqual(out[0],
                         'res',
                         """res output not returned""")

