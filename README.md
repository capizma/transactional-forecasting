# payments-prophet
Payments forecasting at scale.

Hyperparameter tuning has always proven to be effective for increasing model accuracy - however this can be expanded upon.
When trying to predict on time-series data, sometimes the dates in question are the most effective indicator of volume - especially in the topic of trends in transactional data.

payments-prophet aims to combine the effects of hyperparameter tuning with a selection of exogenous regressors - picking the most optimal patterns in dates to use as a basis to futher increase model accuracy.

Model process:

  - Import a .csv file (ideally with two columns, 'reporting-date' in format YYYY-MM-DD and 'volume' - an integer value representing what needs to be predicted.
    Please note - the following steps may take some time if you are uploading multiple files!

  - Run produce_regressors.py. A list of the most suitable regressors for each datafile - identified from using a basic version of Prophet will be output to ./params/hyperfits/, along with their produced sMAPE scores.

  - Run prodce_hyperparams.py. A list of the most suitable parameters for each datafile - idenfitied from using the previously produced regressors from ./params/hyperfits/, will be exported to ./params/hyperparams/

  - If you wish to see a final 3-month cross-validation of any models produced, you can run run_daily_testing.py - this will produce visualisations for any validations in ./results/fit_images/.
    cv_results.csv will also be exported to ./results/ containing all datasets along with their newly evaluated sMAPES (validated against the final 3-month cutoff, these not validated as extensively as any scores from ./params/)
	
  - Run run_daily_forecasts.py. Three months will be forecasted for each daily dataset - with daily and monthly results output to ./results/daily_results/ and ./results/monthly_results_agg/ respectively.
    See the forecasts in ./results/predict_images/!