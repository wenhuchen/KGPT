FOLDER=$(pwd)
GPUS=$1
percent=$2
epoch=$3

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset webnlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 256 --max_dec_len 72 --num_workers 8 --epochs ${epoch} \
 --printing_steps 200 --save_every_n_epochs 40 --encoder sequence --option few_shot --percent ${percent} \
 --learning_rate 5e-5
