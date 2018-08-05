"""Base classes for HTML viz outputs classes"""

# Author: Jason Lewris <jlewris@deloitte.com>
# License: MIT

import pandas as pd

class HtmlFmtMixin(object):
    """Mixin class for formatting result outputs for HTML"""

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