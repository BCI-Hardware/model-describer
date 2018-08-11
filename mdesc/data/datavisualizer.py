import pkg_resources
import re


class DataVisualizer(object):

    # strip long integer rep
    strip_L = re.compile('(?P<digit>[0-9])L')

    def __init__(self, round_num=4):
        self.round_num = round_num

    def main_to_json(self):
        """
        Convert results df into json format for HTML

        input example
        ---------
        MSE  Total  dtype  errNeg  errPos groupByValue groupByVarName  \
        0.0100    1.0  Continuous     NaN     0.1         high        alcohol
        0.2700    4.0  Continuous     0.0     0.6         high        alcohol
        0.0867    6.0  Continuous    -0.2     0.1         high        alcohol

        predictedYSmooth  x_name  x_value
        7.0  volatile acidity    0.115
        7.5  volatile acidity    0.160
        6.5  volatile acidity    0.170

        output example
        ---------
        {'Type': 'Continuous',
        'Data': [{'errPos': 'None', 'chlorides': 0.01, 'groupByVarName': 'alcohol',
        'errNeg': -0.5, 'predictedYSmooth': 5.0, 'groupByValue': 'high'}]}

        :return:
        """

        if self.class_type == 'error':
            to_drop = ['x_name', 'MSE', 'Total', 'dtype', 'std_change']
            groupby = 'x_name'
        else:
            to_drop = ['x_name', 'MSE', 'Total', 'dtype', 'std_change', 'errPos', 'errNeg']
            groupby = ['x_name', 'std_change']

        t = (self._results.fillna('null')
             .round(decimals=4)
             .groupby(groupby)
             .apply(lambda x: (x.rename(columns={'x_value': x['x_name'].unique()[0]})
                               .drop(to_drop, axis=1)
                               .to_dict(orient='rows')))
             .reset_index(name='json_values'))

        if self.class_type == 'error':
            t_list = [{'Data': l} for l in t['json_values'].tolist()]
        else:
            t_list = [{'Data': l, 'Change': change} for l, change in
                      zip(t['json_values'].tolist(), t['std_change'].tolist())]
        tmp = []
        # construct as Continuous
        for d in t_list:
            d['Type'] = 'Continuous'
            tmp.append(d)

        return tmp

    def acc_to_json(self):
        """

        Input Example
        ----------
        ErrType       MSE  Total      Type     Yvar groupByVarName groupbyValue
        MSE  0.095758    356  Accuracy  quality        alcohol         high
        MSE  0.053502   3295  Accuracy  quality        alcohol          low
        MSE  0.084403   2846  Accuracy  quality        alcohol       medium

        Output Example
        -----------
        {'Yvar': 'quality', 'ErrType': 'MSE', 'Type': 'Accuracy',
        'Data': [{'groupbyValue': 'high', 'MSE': 0.09575842696629212, 'Total': 356L...}

        :return:
        """
        # MAIN ACCURACY
        acc = self.accuracy.fillna('null')
        acc = (acc.groupby(['Yvar', 'ErrType', 'Type'])
               .apply(lambda x: x.drop(['Yvar', 'ErrType', 'Type'], axis=1)
                      .to_dict(orient='rows'))
               .reset_index(name='accContainer'))

        out = {}
        out['Data'] = acc['accContainer'].values[0]
        out['Yvar'] = acc['Yvar'].values[0]
        out['ErrType'] = acc['ErrType'].values[0]
        out['Type'] = acc['Type'].values[0]

        return out

    def p_group_to_json(self, p_group_df):
        """

        :param p_group_df: pd.DataFrame - required
            percentiles within variables at specific group levels
        :return:

        Input Example
        -----------
        value percentile groupByVar           colname
        0.115         0%       high  volatile acidity
        0.160         1%       high  volatile acidity
        0.220        10%       high  volatile acidity

        Output Example:
        -----------
        {'Data': [{'variable': 'chlorides', 'percentileList': [{'percentileValues': [{'percentiles': '0%',
        'value': 0.009}, {'percentiles': '1%', 'value': 0.021}, {'percentiles': '10%', 'value': 0.031},
        {'percentiles': '25%', 'value': 0.038}, {'percentiles': '50%', 'value': 0.047},
        {'percentiles': '75%', 'value': 0.065}, {'percentiles': '90%', 'value': 0.086},
        {'percentiles': '100%', 'value': 0.611}], 'groupByVar': 1L}, {'percentileValues':
        [{'percentiles': '0%', 'value': 0.012}, {'percentiles': '1%', 'value': 0.043}, {'percentiles': '10%',
        'value': 0.06}, {'percentiles': '25%', 'value': 0.07}, {'percentiles': '50%', 'value': 0.079},
        {'percentiles': '75%', 'value': 0.09}, {'percentiles': '90%', 'value': 0.109}, ...
        """
        # base layer nesting
        p_group_df = p_group_df.fillna('null')
        tmp = (p_group_df.groupby(['colname', 'groupByVar'])
               .apply(lambda x: x.drop(['groupByVar', 'colname'], axis=1).to_dict(orient='rows'))
               .reset_index(name='percentileValues')
               )
        # second level nesting
        tmp2 = (tmp.groupby('colname')
                .apply(lambda x: x.drop('colname', axis=1).to_dict(orient='rows'))
                .reset_index(name='percentileList')
                .rename(columns={'colname': 'variable'})
                )
        tmp_json = tmp2.to_dict(orient='rows')
        final_dict = {'Type': 'PercentileGroup', 'Data': tmp_json}
        return final_dict

    def to_html(self, fpath=None):
        """save json results to html for dashboard"""

        if fpath is None:
            raise ValueError("""Must have valid fpath, cannot be left None""")

        # format aggregate analysis results
        agg = self.main_to_json()
        acc = self.acc_to_json()
        p_group_json = self.p_group_to_json(self.p_group_df)
        # append percentile list
        agg.append(self.cont_percentile_json)
        agg.append(p_group_json)
        agg.append(acc)

        datastring = str(agg)
        datastring = DataVisualizer.strip_L.sub('\g<digit>', datastring)

        #html_path = pkg_resources.resource_filename('mdesc', '{}.txt'.format('html_error'))
        html_path = r'mdesc/data/html/html_error.txt'
        html = open(html_path, 'r').read()
        output = html.replace('<***>', datastring)
        output = output.replace("'", '"')

        with open(fpath, 'w') as outfile:
            outfile.write(output)

    def to_viz(self):
        pass
