import tensorflow as tf
import numpy as np
import os
import time
import datetime
import operator
from collections import defaultdict
from model import metrics
from model import data_helpers
from model.model_IMN import IMN

# Files
tf.flags.DEFINE_string("train_file", "", "path to train file")
tf.flags.DEFINE_string("valid_file", "", "path to valid file")
tf.flags.DEFINE_string("response_file", "", "path to response file")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("char_vocab_file",  "", "path to char vocab file")
tf.flags.DEFINE_string("embedded_vector_file", "", "pre-trained embedded word vector")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_utter_len", 50, "max utterance length")
tf.flags.DEFINE_integer("max_utter_num", 10, "max utterance number")
tf.flags.DEFINE_integer("max_response_len", 50, "max response length")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")
tf.flags.DEFINE_integer("num_layer", 3, "max response length")
tf.flags.DEFINE_integer("embedding_dim", 200, "dimensionality of word embedding")
tf.flags.DEFINE_integer("rnn_size", 200, "number of RNN units")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "batch size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout keep probability (default: 1.0)")
tf.flags.DEFINE_integer("num_epochs", 1000000, "number of training epochs (default: 1000000)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "evaluate model on valid dataset after this many steps (default: 1000)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")

vocab = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))
charVocab = data_helpers.load_char_vocab(FLAGS.char_vocab_file)

response_data = data_helpers.load_responses(FLAGS.response_file, vocab, FLAGS.max_response_len)
print('response_data size: {}'.format(len(response_data)))

train_dataset = data_helpers.load_dataset(FLAGS.train_file, vocab, FLAGS.max_utter_len, FLAGS.max_utter_num, response_data)
print('train_pairs: {}'.format(len(train_dataset)))
valid_dataset = data_helpers.load_dataset(FLAGS.valid_file, vocab, FLAGS.max_utter_len, FLAGS.max_utter_num, response_data)
print('valid_pairs: {}'.format(len(valid_dataset)))

target_loss_weight=[1.0,1.0]

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        imn = IMN(
            max_utter_len=FLAGS.max_utter_len,
            max_utter_num=FLAGS.max_utter_num,
            max_response_len=FLAGS.max_response_len,
            num_layer=FLAGS.num_layer,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            vocab=vocab,
            rnn_size=FLAGS.rnn_size,
            maxWordLength=FLAGS.max_word_length,
            charVocab=charVocab,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                   5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(imn.mean_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        """
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        """

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        """
        loss_summary = tf.scalar_summary("loss", imn.mean_loss)
        acc_summary = tf.scalar_summary("accuracy", imn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        """

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_utterances, x_response, x_utterances_len, x_response_len, 
                       x_utters_num, x_target, x_target_weight, id_pairs,
                       x_u_char, x_u_char_len, x_r_char, x_r_char_len):
            """
            A single training step
            """
            feed_dict = {
              imn.utterances: x_utterances,
              imn.response: x_response,
              imn.utterances_len: x_utterances_len,
              imn.response_len: x_response_len,
              imn.utters_num: x_utters_num,
              imn.target: x_target,
              imn.target_loss_weight: x_target_weight,
              imn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              imn.u_charVec: x_u_char,
              imn.u_charLen: x_u_char_len,
              imn.r_charVec: x_r_char,
              imn.r_charLen: x_r_char_len
            }

            _, step, loss, accuracy, predicted_prob = sess.run(
                [train_op, global_step, imn.mean_loss, imn.accuracy, imn.probs],
                feed_dict)

            if step%100 == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)


        def dev_step():
            results = defaultdict(list)
            num_test = 0
            num_correct = 0.0
            valid_batches = data_helpers.batch_iter(valid_dataset, FLAGS.batch_size, 1, target_loss_weight, FLAGS.max_utter_len, FLAGS.max_utter_num, FLAGS.max_response_len, charVocab, FLAGS.max_word_length, shuffle=True)
            for valid_batch in valid_batches:
                x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs, x_u_char, x_u_char_len, x_r_char, x_r_char_len = valid_batch
                feed_dict = {
                  imn.utterances: x_utterances,
                  imn.response: x_response,
                  imn.utterances_len: x_utterances_len,
                  imn.response_len: x_response_len,
                  imn.utters_num: x_utters_num,
                  imn.target: x_target,
                  imn.target_loss_weight: x_target_weight,
                  imn.dropout_keep_prob: 1.0,
                  imn.u_charVec: x_u_char,
                  imn.u_charLen: x_u_char_len,
                  imn.r_charVec: x_r_char,
                  imn.r_charLen: x_r_char_len,
                }
                batch_accuracy, predicted_prob = sess.run([imn.accuracy, imn.probs], feed_dict)
                num_test += len(predicted_prob)
                if num_test % 1000 == 0:
                    print(num_test)

                num_correct += len(predicted_prob) * batch_accuracy
                for i, prob_score in enumerate(predicted_prob):
                    question_id, response_id, label = id_pairs[i]
                    results[question_id].append((response_id, label, prob_score))

            #calculate top-1 precision
            print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct/num_test))
            accu, precision, recall, f1, loss = metrics.classification_metrics(results)
            print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

            mvp = metrics.mean_average_precision(results)
            mrr = metrics.mean_reciprocal_rank(results)
            top_1_precision = metrics.top_1_precision(results)
            total_valid_query = metrics.get_num_valid_query(results)
            print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

            return mrr

        best_mrr = 0.0
        batches = data_helpers.batch_iter(train_dataset, FLAGS.batch_size, FLAGS.num_epochs, target_loss_weight, FLAGS.max_utter_len, FLAGS.max_utter_num, FLAGS.max_response_len, charVocab, FLAGS.max_word_length, shuffle=True)
        for batch in batches:
            x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs, x_u_char, x_u_char_len, x_r_char, x_r_char_len = batch
            train_step(x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs, x_u_char, x_u_char_len, x_r_char, x_r_char_len)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                valid_mrr = dev_step()
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

