FOLDER=$(pwd)
GPUS=$1

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset e2enlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 325 --max_dec_len 72 --num_workers 4 --epochs 50 \
 --printing_steps 50 --save_every_n_epochs 10  --encoder graph
