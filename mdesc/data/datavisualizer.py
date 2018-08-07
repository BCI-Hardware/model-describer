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
        t = (self._results.fillna('None')
             .round(decimals=2)
             .groupby('x_name')
             .apply(lambda x: (x.rename(columns={'x_value': x['x_name'].unique()[0]})
                               .drop(['x_name', 'MSE', 'Total', 'dtype'], axis=1)
                               .to_dict(orient='rows')))
             .reset_index(name='json_values'))

        t_list = [{'Data': l} for l in t['json_values'].tolist()]
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
        acc = (self.accuracy.groupby(['Yvar', 'ErrType', 'Type'])
               .apply(lambda x: x.drop(['Yvar', 'ErrType', 'Type'], axis=1)
                      .to_dict(orient='rows'))
               .reset_index(name='accContainer'))

        out = {}
        out['Data'] = acc['accContainer'].values[0]
        out['Yvar'] = acc['Yvar'].values[0]
        out['ErrType'] = acc['ErrType'].values[0]
        out['Type'] = acc['Type'].values[0]

        return out

    def to_html(self, fpath=None):
        """save json results to html for dashboard"""

        if fpath is None:
            raise ValueError("""Must have valid fpath, cannot be left None""")

        # format aggregate analysis results
        agg = self.main_to_json()
        acc = self.acc_to_json()
        # append percentile list
        agg.extend(self.percentile_list)
        agg.append(acc)
        datastring = str(agg)
        datastring = DataVisualizer.strip_L.sub('\g<digit>', datastring)

        # load correct HTML format
        html_path = pkg_resources.resource_filename('mdesc', '{}.txt'.format('html_error'))
        html = open(html_path, 'r').read()
        output = html.replace('<***>', datastring)

        with open(fpath, 'w') as outfile:
            outfile.write(output)

    def to_viz(self):
        pass
