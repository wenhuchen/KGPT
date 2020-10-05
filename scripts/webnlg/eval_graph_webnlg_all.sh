FOLDER=$(pwd)
GPUS=$1
OPTION=$2
checkpoint=$3

for file in ${FOLDER}/${checkpoint}/*.pt
do
    echo "Evaluation on", ${file}
    CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 32 --dataset webnlg \
     --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 325 --max_dec_len 72 --num_workers 0 --option ${OPTION} \
     --load_from ${file} --beam_size 2 --encoder graph
done
