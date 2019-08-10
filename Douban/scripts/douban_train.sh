cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data/Douban_Corpus

train_file=$DATA_DIR/train.txt
valid_file=$DATA_DIR/valid.txt
test_file=$DATA_DIR/test.txt
response_file=$DATA_DIR/responses.txt
vocab_file=$DATA_DIR/vocab.txt
embedded_vector_file=$DATA_DIR/tencent_200_plus_word2vec_200.txt

max_utter_len=50
max_utter_num=10
max_response_len=50
num_layer=3
DIM=400
rnn_size=200

batch_size=128
lambda=0
dropout_keep_prob=1
num_epochs=10
evaluate_every=1000

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python -u ${PKG_DIR}/model/train.py \
                --train_file $train_file \
                --valid_file $valid_file \
                --test_file $test_file \
                --response_file $response_file \
                --vocab_file $vocab_file \
                --embeded_vector_file $embedded_vector_file \
                --max_utter_len $max_utter_len \
                --max_utter_num $max_utter_num \
                --max_response_len $max_response_len \
                --num_layer $num_layer \
                --embedding_dim $DIM \
                --rnn_size $rnn_size \
                --batch_size $batch_size \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --num_epochs $num_epochs \
                --evaluate_every $evaluate_every > log_train_IMN_Douban.txt 2>&1 &
