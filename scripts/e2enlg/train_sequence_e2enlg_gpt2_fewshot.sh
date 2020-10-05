FOLDER='/data/wenhu/entity2text'
GPUS=$1
percent=$2

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset e2enlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 256 --max_dec_len 72 --num_workers 8 --epochs 300 \
 --printing_steps 200 --save_every_n_epochs 100 --additional _gated --gated --option few_shot --percent ${percent} \
