FOLDER='/data/wenhu/entity2text'
GPUS=$1
RESUME=$2
percent=$3

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset e2enlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 256 --max_dec_len 72 --num_workers 8 --epochs 100 \
 --printing_steps 200 --save_every_n_epochs 20 --learning_rate 2e-5 --finetune --gated \
 --load_from $RESUME --option few_shot --percent ${percent}
