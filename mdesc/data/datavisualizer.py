import pkg_resources
import re
import random

import numpy as np
from scipy import signal

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go


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


class NotebookVisualizer(DataVisualizer):
    rgb = lambda: random.randint(0, 255)

    def __init__(self, rgb=None, round_num=4):
        if rgb is None:
            self.r, self.g, self.b = [NotebookVisualizer.rgb() for i in range(3)]
        else:
            self.r, self.g, self.b = rgb

        self.traces = []
        self.all_indices = []
        self.all_x_names = []
        self.col_lookup = {}
        self.layout_lookup = {}
        self.range_lookup = {}
        self.title_lookup = {}
        super(NotebookVisualizer, self).__init__(round_num=round_num)

    def smooth_line(self, input):
        if self.smooth is True:
            input = signal.savgol_filter(input, 53, 3)
        return input

    def _create_base_trace(self, level, x, y, rgb=None):

        if rgb is None:
            r, g, b = self.r, self.g, self.b
        else:
            r, g, b = rgb

        trace1 = go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb({},{},{})'.format(r, g, b), shape='spline'),
            mode='lines',
            name=level,
        )

        return trace1

    def _create_error_trace(self, level, x, x_rev, y, y_upper, y_lower,
                            rgb=None):
        # pick colors
        if rgb is None:
            r, g, b = self.r, self.g, self.b
        else:
            r, g, b = rgb
        trace1 = go.Scatter(
            x=x + x_rev,
            y=y_upper + y_lower,
            fill='tozerox',
            fillcolor='rgba({},{},{},0.2)'.format(r, g, b),
            line=dict(color='rgba(255,255,255,0)', shape='spline'),
            showlegend=False,
            name=level,
        )

        trace2 = self._create_base_trace(level, x, y, rgb=rgb)

        return (trace1, trace2)

    def _base_layer(self, df=None):

        for x_idx, x_name in enumerate(df['x_name'].unique()):
            x_group_slice = df.loc[(df['groupByVarName'] == self.groupby_name) & (df['x_name'] == x_name)]
            master_x = x_group_slice['x_value'].values
            self.all_x_names.append(x_name)
            self.col_lookup[x_name] = []
            self.range_lookup[x_name] = [np.min(master_x), np.max(master_x)]
            self.title_lookup[x_name] = '{} for {} {} chart'.format(x_name, self.groupby_name, self.class_type)
            yield (x_name, x_group_slice)

    def _create_nested_error_traces(self, df=None,
                                    groupby_name=None):
        """
        Create plotly traces for each feature and level within specified groupby_var

        """
        for x_name, x_group_slice in self._base_layer(df=df):

            for l_idx, level in enumerate(self.levels):
                mask = x_group_slice['groupByValue'] == level
                y = x_group_slice.loc[mask, 'predictedYSmooth'].values
                y_lower = (y + x_group_slice.loc[mask, 'errNeg'].values).tolist()
                y_lower = y_lower[::-1]
                y_upper = (y + x_group_slice.loc[mask, 'errPos'].values).tolist()
                x = x_group_slice.loc[mask, 'x_value'].tolist()
                x_rev = x[::-1]

                trace1, trace2 = self._create_error_trace(level, x, x_rev, y, y_upper, y_lower,
                                                          rgb=self.rgb_anchor[l_idx])
                trace_tracker_1 = len(self.traces)
                trace_tracker_2 = trace_tracker_1 + 1
                self.traces.append(trace1)
                self.all_indices.append(trace_tracker_1)
                self.traces.append(trace2)
                self.all_indices.append(trace_tracker_2)
                self.col_lookup[x_name].append(trace_tracker_1)
                self.col_lookup[x_name].append(trace_tracker_2)

    def _create_nested_sensitivity_traces(self, df=None):

        for x_name, x_group_slice in self._base_layer(df=df):

            for l_idx, level in enumerate(self.levels):
                mask = x_group_slice['groupByValue'] == level
                y = x_group_slice.loc[mask, 'predictedYSmooth'].values
                x = x_group_slice.loc[mask, 'x_value'].tolist()

                trace1 = self._create_base_trace(level, x, y, rgb=self.rgb_anchor[l_idx])
                trace_tracker_1 = len(self.traces)
                self.traces.append(trace1)
                self.all_indices.append(trace_tracker_1)
                self.col_lookup[x_name].append(trace_tracker_1)

    def _create_update_menu(self, buttons):

        updatemenus = list([
            dict(active=-1,
                 buttons=buttons,
                 direction='down',
                 pad={'r': 10, 't': 10},
                 showactive=True,
                 xanchor='left',
                 yanchor='top',
                 y=1.12,
                 x=1,
                 )
        ])

        return updatemenus

    def _create_buttons(self):

        buttons = []

        for x_name in self.all_x_names:
            format_dict = {}
            format_dict['label'] = x_name
            format_dict['method'] = 'update'
            format_dict['args'] = [
                {'visible': np.isin(self.all_indices, self.col_lookup[x_name])},
                {'title': '{} VIZ {}; Groupby: {}'.format(self.class_type, x_name, self.groupby_name),
                 'xaxis': dict(
                     gridcolor='rgb(255,255,255)',
                     range=self.range_lookup[x_name],
                     showgrid=True,
                     showline=False,
                     showticklabels=True,
                     tickcolor='rgb(127,127,127)',
                     ticks='outside',
                     zeroline=False,
                     title=x_name
                 )}
            ]
            buttons.append(format_dict)

        return buttons

    def _create_layout(self, updatemenus):
        if self.class_type == 'sensitivity':
            title = 'Predicted impact on {}'.format(self.target_name)
        else:
            title = 'Predicted {}'.format(self.target_name)
        layout = go.Layout(
            title='{} Visualization Chart'.format(self.class_type),
            paper_bgcolor='rgb(255,255,255)',
            plot_bgcolor='rgb(229,229,229)',
            xaxis=dict(
                gridcolor='rgb(255,255,255)',
                # range=[np.min(x), np.max(x)],
                showgrid=True,
                showline=False,
                showticklabels=True,
                tickcolor='rgb(127,127,127)',
                ticks='outside',
                zeroline=False,
                title='x axis'
            ),
            yaxis=dict(
                gridcolor='rgb(255,255,255)',
                showgrid=True,
                showline=False,
                showticklabels=True,
                tickcolor='rgb(127,127,127)',
                ticks='outside',
                zeroline=False,
                title=title
            ),
            updatemenus=updatemenus,
        )

        return layout

    def viz_now(self, groupby_name=None, rgb=None,
                smooth=True):

        if groupby_name is None:
            raise ValueError("""Must select viable groupby variable. Available groupby variables
            inclue: {}""".format(None))  # TODO add self.groupby_df names

        self.groupby_name = groupby_name
        self.smooth = smooth

        # iterate over levels within groupby_name
        self.levels = self._results.loc[self._results['groupByVarName'] == self.groupby_name, 'groupByValue'].unique()
        rgb = NotebookVisualizer.rgb
        self.rgb_anchor = [(rgb(), rgb(), rgb()) for level in self.levels]

        if self.class_type == 'sensitivity':

            self._create_nested_sensitivity_traces(df=self._results)
        else:
            self._create_nested_error_traces(df=self._results)

        buttons = self._create_buttons()

        updatemenus = self._create_update_menu(buttons)

        layout = self._create_layout(updatemenus)

        # layout['updatemenus'] = updatemenus

        data = self.traces
        fig = go.Figure(data=data, layout=layout)
        return iplot(fig)

