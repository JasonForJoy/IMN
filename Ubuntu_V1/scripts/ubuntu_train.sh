cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data/Ubuntu_Corpus_V1

train_file=$DATA_DIR/train.txt
valid_file=$DATA_DIR/valid.txt
response_file=$DATA_DIR/responses.txt
vocab_file=$DATA_DIR/vocab.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
embedded_vector_file=$DATA_DIR/glove_42B_300d_vec_plus_word2vec_100.txt

max_utter_len=50
max_utter_num=10
max_response_len=50
max_word_length=18
num_layer=3
embedding_dim=400
rnn_size=200

batch_size=96
lambda=0
dropout_keep_prob=0.8
num_epochs=10
evaluate_every=1000

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=3 python -u ${PKG_DIR}/model/train.py \
                --train_file $train_file \
                --valid_file $valid_file \
                --response_file $response_file \
                --vocab_file $vocab_file \
                --char_vocab_file $char_vocab_file \
                --embedded_vector_file $embedded_vector_file \
                --max_utter_len $max_utter_len \
                --max_utter_num $max_utter_num \
                --max_response_len $max_response_len \
                --max_word_length $max_word_length \
                --num_layer $num_layer \
                --embedding_dim $embedding_dim \
                --rnn_size $rnn_size \
                --batch_size $batch_size \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --num_epochs $num_epochs \
                --evaluate_every $evaluate_every > log_train_IMN_UbuntuV1.txt 2>&1 &
