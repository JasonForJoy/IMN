import numpy as np
import random


def load_char_vocab(fname):
    '''
    charVocab = {"U": 0, "!": 1, ...}
    '''
    charVocab={}
    with open(fname, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            char_id = int(fields[0])
            ch = fields[1]
            charVocab[ch] = char_id
    return charVocab

def load_vocab(fname):
    '''
    vocab = {"<PAD>": 0, ...}
    '''
    vocab={}
    with open(fname, 'rt') as f:
        for line in f:
            fields = line.decode('utf-8').strip().split('\t')
            vocab[fields[0]] = int(fields[1])
    return vocab

def to_vec(tokens, vocab, maxlen):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["unknown"])

    return length, np.array(vec)

def load_responses(fname, vocab, maxlen):
    responses={}
    with open(fname, 'rt') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            fields = line.split('\t')
            if len(fields) != 2:
                print("WRONG LINE: {}".format(line))
                r_text = 'unknown'
            else:
                r_text = fields[1]
            tokens = r_text.split(' ')[:maxlen]
            len1, vec = to_vec(tokens, vocab, maxlen)
            responses[fields[0]] = (len1, vec, tokens)
    return responses

def load_dataset(fname, vocab, max_utter_len, max_utter_num, responses):

    dataset=[]
    with open(fname, 'rt') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            fields = line.split('\t')
            us_id = fields[0]

            context = fields[1]
            utterances = context.split(" __EOS__ ")
            new_utterances = []
            for utterance in utterances:
                new_utterances.append(utterance + " __EOS__")
            utterances = new_utterances[-max_utter_num:]  # select the last max_utter_num utterances

            us_tokens = []
            us_vec = []
            us_len = []
            for utterance in utterances:
                u_tokens = utterance.split(' ')[:max_utter_len]  # select the first max_utter_len tokens in every utterance
                u_len, u_vec = to_vec(u_tokens, vocab, max_utter_len)
                us_tokens.append(u_tokens)
                us_vec.append(u_vec)
                us_len.append(u_len)

            us_num = len(utterances)

            if fields[3] != "NA":
                neg_ids = [id for id in fields[3].split('|')]
                for r_id in neg_ids:
                    r_len, r_vec, r_tokens = responses[r_id]
                    dataset.append((us_id, us_len, us_vec, us_num, r_id, r_len, r_vec, 0.0, us_tokens, r_tokens))
                    # break  # uncomment this line when testing recall_2@1

            if fields[2] != "NA":
                pos_ids = [id for id in fields[2].split('|')]
                for r_id in pos_ids:
                    r_len, r_vec, r_tokens = responses[r_id]
                    dataset.append((us_id, us_len, us_vec, us_num, r_id, r_len, r_vec, 1.0, us_tokens, r_tokens))
    return dataset


def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec


def charVec(tokens, charVocab, maxlen, maxWordLength):
    '''
    chars = np.array( (maxlen, maxWordLength) )    0 if not found in charVocab or None
    word_lengths = np.array( maxlen )              1 if None
    '''
    n = len(tokens)
    if n > maxlen:
        n = maxlen

    chars =  np.zeros((maxlen, maxWordLength), dtype=np.int32)
    word_lengths = np.ones(maxlen, dtype=np.int32)
    for i in range(n):
        token = tokens[i][:maxWordLength]
        word_lengths[i] = len(token)
        row = chars[i]
        for idx, ch in enumerate(token):
            if ch in charVocab:
                row[idx] = charVocab[ch]

    return chars, word_lengths


def batch_iter(data, batch_size, num_epochs, target_loss_weights, max_utter_len, max_utter_num, max_response_len, charVocab, max_word_length, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_utterances = []
            x_response = []
            x_utterances_len = []
            x_response_len = []
            targets = []
            target_weights=[]
            id_pairs =[]

            x_utterances_char=[]
            x_utterances_char_len=[]
            x_response_char=[]
            x_response_char_len=[]

            x_utterances_num = []

            for rowIdx in range(start_index, end_index):
                us_id, us_len, us_vec, us_num, r_id, r_len, r_vec, label, us_tokens, r_tokens = data[rowIdx]
                if label > 0:
                    target_weights.append(target_loss_weights[1])
                else:
                    target_weights.append(target_loss_weights[0])

                # normalize us_vec and us_len
                new_utters_vec = np.zeros((max_utter_num, max_utter_len), dtype='int32')
                new_utters_len = np.zeros((max_utter_num, ), dtype='int32')
                for i in range(len(us_len)):
                    new_utter_vec = normalize_vec(us_vec[i], max_utter_len)
                    new_utters_vec[i] = new_utter_vec
                    new_utters_len[i] = us_len[i]
                new_r_vec = normalize_vec(r_vec, max_response_len)

                x_utterances.append(new_utters_vec)
                x_utterances_len.append(new_utters_len)
                x_response.append(new_r_vec)
                x_response_len.append(r_len)
                targets.append(label)
                id_pairs.append((us_id, r_id, int(label)))

                # normalize CharVec and CharLen
                uttersCharVec = np.zeros((max_utter_num, max_utter_len, max_word_length), dtype='int32')
                uttersCharLen = np.ones((max_utter_num, max_utter_len), dtype='int32')
                for i in range(len(us_len)):
                    utterCharVec, utterCharLen = charVec(us_tokens[i], charVocab, max_utter_len, max_word_length)
                    uttersCharVec[i] = utterCharVec
                    uttersCharLen[i] = utterCharLen
                rCharVec, rCharLen = charVec(r_tokens, charVocab, max_response_len, max_word_length)

                x_utterances_char.append(uttersCharVec)
                x_utterances_char_len.append(uttersCharLen)
                x_response_char.append(rCharVec)
                x_response_char_len.append(rCharLen)

                x_utterances_num.append(us_num)

            yield np.array(x_utterances), np.array(x_response), np.array(x_utterances_len), np.array(x_response_len), \
                  np.array(x_utterances_num), np.array(targets), np.array(target_weights), id_pairs, \
                  np.array(x_utterances_char), np.array(x_utterances_char_len), np.array(x_response_char), np.array(x_response_char_len)

