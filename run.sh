CURRENT_DIR=`pwd`
python run.py \
  --do_train=1 \
  --do_predict=0 \
  --file_name_or_path=$CURRENT_DIR/data/ \
  --train_file_path=$CURRENT_DIR/data/ \
  --valid_file_path=$CURRENT_DIR/data/ \
  --predict_file_path=$CURRENT_DIR/data/ \
  --model_name_or_path=$CURRENT_DIR/model/ \
  --output_dir=$CURRENT_DIR/outputs/ \
  --predict_model_name_or_path=$CURRENT_DIR/outputs/model_checkpoint20.0/checkpoint-20000
