GPU=0
LOG_FILE=./experiments${GPU}.out
MODELS=('cnn' 'mlp' 'lstm')
MODELS_ML=('rf' 'lr')
METRICS=('mse' 'mae' 'rmse' 'std' 'std_diff')
PARAMETERS=./parameters_full2.json
OUTPUT=results
CSV_FILENAME=results.csv

python main.py --models ${MODELS[@]} --models_ml ${MODELS_ML[@]} --gpu ${GPU}  --metrics ${METRICS[@]} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
