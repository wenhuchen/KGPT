FOLDER=$(pwd)
GPUS=$1
percent=$2

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset wikibionlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 425 --max_dec_len 72 --num_workers 8 --epochs 400 \
 --printing_steps 200 --save_every_n_epochs 100  --encoder sequence --finetune --learning_rate 1e-5 \
 --load_from $FOLDER/checkpoint_wikidata_full/checkpoint_sequence_head8_layer5_GPT2_maxfact12_lower/model_ep11.pt \
 --option few_shot --percent ${percent}
