.. -*- mode: rst -*-

Universal WhiteBox Parameters
======================

modelobj : fitted model, required
------------
WhiteBox is designed first and foremost to work with sklearn. Anything passed as modelobj must be a single sklearn object that has been fit prior to being passed to WhiteBox. 

model_df : pandas DataFrame, required
------------
The data used for modeling (both continuous and dummy variables). This data frame must contain all of the 'X' columns and the 'y' column used in your modelobject.fit(X, y) modeling step. Dummy variables formed from categorical variables must have the form catvarname_value (Gender_Male, Gender_Female, etc.) or they will cause and error and not map to the output charts. 

For dummies, all values in a categorical column (including nulls) must have a dummy column associated with them. The following code will create dummies in the correct format.

.. code-block:: python
  
  #create dummies
  dummies = pd.concat([pd.get_dummies(model_df.loc[:, col], prefix = col) for col in model_df.select_dtypes(include =   
  ['object']).columns], dummy_na=True, axis = 1)

ydepend : string, required
------------
The dependent 'y' variable you are trying to predict. Dependent variables can be continuous or binary. 

cat_df : pandas DataFrame, required
------------
DataFrame of variables with the object data type. This dataframe may contain string variables not present in the model. The groupbyvars must be contained in this dataset. The dependent variable (ydpend) must also be included in this dataset.  

groupbyvars : List, required
------------

List of variables that 'groups' the output into discrete segments for comparison. If the user would like to see the entire population without subgroups, just add a variable to cat_df that only has one category (such as 'Entire Population') and add that variable to the groupbyvars list. In this special case, it must be the only variable in the groupbyvars list.   

.. code-block:: python

  #add uniform
  cat_df['Entire_Population'] = 'Entire Population'

keepfeaturelist  : list, optional
------------

The items in this list limit the output. Only variables present in the list will display in the final HTML output. 

Note- Groupbyvars must be listed if featuredict is specified.

verbose : int, optional 
-------------
Logging level of output. Level -- 0 = debug, 1 = warning, 2 = error.

ErrorViz Specific Parameter
=======================

error_type : string, optional
---------------------

Aggregate error metric that summarizes the positive and negative error vectors. It can take the values: 'MSE' (mean squared error), 'MAE' (mean absolute error), or 'RMSE' (root mean squared error). By default, it is the MAE so errors of [-2,-1,3,4,5] would result in an average negative error of (2+1)/2 and an average positive error of (3+4+5)/3. 

SensitivityViz Specific Parameter
=======================

std_num : float, optional
Number of standard deviations to push data for syntehtic variable creation in the sensitivity analysis. Larger values will result in larger 'leaps of faith' of the model where it will be pushing more data outside of the range of observed data. Only values between -3 and 3 will be accepted as it is generally unrealistic to change variables more than 3 standard deviations. 

