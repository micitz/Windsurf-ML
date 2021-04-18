# Windsurf-ML
Calibrate Windsurf and produce forecasts

## Neural Network Calibration.py
Take previous simulationsand train a NN to predict the RMSE from the input parameters. Then run a genetic algorithm with the trained NN to identify the (potential) best parameterization for the hindcast

## windsurf_ga.py / Windsurf.py
Run a genetic algorithm with Windsurf starting with the prediction outputted from `Neural Network Calibration.py`

## Genetic Algorithm Analysis.py / Windsurf_GA_Result.py
Analyze output from the Windsurf GA simulations

## Neural Network Forecasts.py
Take the Windsurf output morphology and train an LSTM on it, then make a forecat using the traiend LSTM
