FOLDER=$(pwd)
GPUS=$1
RESUME=$2

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset e2enlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 325 --max_dec_len 72 --num_workers 4 --epochs 30 \
 --printing_steps 200 --save_every_n_epochs 2 --encoder graph --finetune --learning_rate 2e-5\
 --load_from ${RESUME}
