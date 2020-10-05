FOLDER=$(pwd)
GPUS=$1
OPTION=$2
file=$3

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64  --dataset wikibionlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 512 --max_dec_len 72 --num_workers 4 --option ${OPTION} \
 --load_from ${file} --encoder graph --beam_size 2
