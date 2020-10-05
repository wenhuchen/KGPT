FOLDER=$(pwd)
GPUS=$1
RESUME=$2
percent=$3

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset webnlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 256 --max_dec_len 72 --num_workers 8 --epochs 1000 \
 --printing_steps 200 --save_every_n_epochs 200 --learning_rate 4e-5 --finetune --encoder sequence \
 --load_from $RESUME --option few_shot --percent ${percent}