import tensorflow as tf
from model import data_helpers
from collections import defaultdict
import operator
import metrics

# Files
tf.flags.DEFINE_string("test_file", "", "path to test file")
tf.flags.DEFINE_string("response_file", "", "path to response file")
tf.flags.DEFINE_string("vocab_file", "", "path to vocabulary file")
tf.flags.DEFINE_string("output_file", "", "prediction output file")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_utter_len", 50, "max utterance length")
tf.flags.DEFINE_integer("max_utter_num", 10, "max utterance number")
tf.flags.DEFINE_integer("max_response_len", 50, "max response length")
tf.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.flags.DEFINE_string("checkpoint_dir", "", "checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

vocab = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))

response_data = data_helpers.load_responses(FLAGS.response_file, vocab, FLAGS.max_response_len)
test_dataset = data_helpers.load_dataset(FLAGS.test_file, vocab, FLAGS.max_utter_len, FLAGS.max_utter_num, response_data)
print('test_pairs: {}'.format(len(test_dataset)))

target_loss_weight=[1.0,1.0]

print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        utterances = graph.get_operation_by_name("utterances").outputs[0]
        response   = graph.get_operation_by_name("response").outputs[0]

        utterances_len = graph.get_operation_by_name("utterances_len").outputs[0]
        response_len = graph.get_operation_by_name("response_len").outputs[0]
        utterances_num = graph.get_operation_by_name("utterances_num").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        prob = graph.get_operation_by_name("prediction_layer/prob").outputs[0]

        results = defaultdict(list)
        num_test = 0
        test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, target_loss_weight, FLAGS.max_utter_len, FLAGS.max_utter_num, FLAGS.max_response_len, shuffle=False)
        for test_batch in test_batches:
            x_utterances, x_response, x_utterances_len, x_response_len, x_utters_num, x_target, x_target_weight, id_pairs = test_batch
            feed_dict = {
                utterances: x_utterances,
                response: x_response,
                utterances_len: x_utterances_len,
                response_len: x_response_len,
                utterances_num: x_utters_num,
                dropout_keep_prob: 1.0
            }
            predicted_prob = sess.run(prob, feed_dict)
            num_test += len(predicted_prob)
            print('num_test_sample={}'.format(num_test))
            for i, prob_score in enumerate(predicted_prob):
                us_id, r_id, label = id_pairs[i]
                results[us_id].append((r_id, label, prob_score))

accu, precision, recall, f1, loss = metrics.classification_metrics(results)
print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

mvp = metrics.mean_average_precision(results)
mrr = metrics.mean_reciprocal_rank(results)
top_1_precision = metrics.top_1_precision(results)
total_valid_query = metrics.get_num_valid_query(results)
print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

out_path = FLAGS.output_file
print("Saving evaluation to {}".format(out_path))
with open(out_path, 'w') as f:
    f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
    for us_id, v in results.items():
        v.sort(key=operator.itemgetter(2), reverse=True)
        for i, rec in enumerate(v):
            r_id, label, prob_score = rec
            rank = i+1
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(us_id, r_id, prob_score, rank, label))
