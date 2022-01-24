GPU=0
LOG_FILE=./experiments${GPU}.out
MODELS=('lstm')
MODELS_ML=('rf')
METRICS=('mse' 'mae' 'rmse')
PARAMETERS=./parameters.json
OUTPUT=results
CSV_FILENAME=results.csv

python main.py --models ${MODELS[@]} --models_ml ${MODELS_ML[@]} --gpu ${GPU}  --metrics ${METRICS[@]} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
