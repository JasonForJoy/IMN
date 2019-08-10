cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data/Ubuntu_Corpus_V1

latest_run=`ls -dt runs/* |head -n 1`
latest_checkpoint=${latest_run}/checkpoints
# latest_checkpoint=runs/1550714951/checkpoints  # or choose the model to test manually
echo $latest_checkpoint

test_file=$DATA_DIR/test.txt
response_file=$DATA_DIR/responses.txt
vocab_file=$DATA_DIR/vocab.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
output_file=./ubuntu_test_out.txt

max_utter_len=50
max_utter_num=10
max_response_len=50
max_word_length=18
batch_size=128

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python -u ${PKG_DIR}/model/eval.py \
                  --test_file $test_file \
                  --response_file $response_file \
                  --vocab_file $vocab_file \
                  --char_vocab_file $char_vocab_file \
                  --output_file $output_file \
                  --max_utter_len $max_utter_len \
                  --max_utter_num $max_utter_num \
                  --max_response_len $max_response_len \
                  --max_word_length $max_word_length \
                  --batch_size $batch_size \
                  --checkpoint_dir $latest_checkpoint > log_test_IMN_UbuntuV1.txt 2>&1 &
