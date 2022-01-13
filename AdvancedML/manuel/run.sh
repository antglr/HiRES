GPU=0
LOG_FILE=./experiments${GPU}.out
MODELS=('lstm' 'cnn' 'mlp' )
MODELS_ML=('rf' 'lr')
METRICS=('mae' 'rmse' 'mse')
PARAMETERS=./parameters_full.json
OUTPUT=results
CSV_FILENAME=results.csv

python main.py --models ${MODELS[@]} --models_ml ${MODELS_ML[@]} --gpu ${GPU}  --metrics ${METRICS[@]} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
