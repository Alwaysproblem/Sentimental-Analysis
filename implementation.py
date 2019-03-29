import tensorflow as tf


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 200  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than',
                  'wouldn', 'shouldn', 'll', 'aren', 'isn', 'get'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    """
    input: the content of each training file. type : string("\n" means return)
    used in the runner file line 58

    output: word list form.

    Note: remeber choice 100 words at random 
    """
    import re

    page = r"<.*?>"
    pieces_nopara = re.compile(page).sub("", review)

    patten = r"\W+"
    pieces = re.compile(patten).split(pieces_nopara)

    piece = [p.lower() for p in pieces if p != '' and p.lower() not in stop_words and len(p) > 2]

    processed_review = piece

    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    """
    training_data_embedded[exampleNum., ]
    input data is placeholder, size NUM_SAMPLES x MAX_WORDS_IN_REVIEW x EMBEDDING_SIZE
    labels placeholder,
    dropout_keep_prob placeholder,
    optimizer is function with placeholder input_data, labels, dropout_keep_prob
    Accuracy, loss is function with placeholder input_data, labels
    """
    lstm_hidden_unit = 256

    learning_rate = 0.00023

    training = tf.placeholder_with_default(False, shape = (), name="IsTraining")

    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=(), name='drop_rate')

    with tf.name_scope("InputData"):
        input_data = tf.placeholder(
            tf.float32, 
            shape=(BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE),
            name="input_data"
        )
    with tf.name_scope("Labels"):
        labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2), name="labels")

    with tf.name_scope("BiRNN"):
        LSTM_cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_unit)
        LSTM_cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_unit)

        LSTM_drop_fw = tf.nn.rnn_cell.DropoutWrapper(
            cell = LSTM_cell_fw, 
            output_keep_prob = dropout_keep_prob
        )

        LSTM_drop_bw = tf.nn.rnn_cell.DropoutWrapper(
            cell = LSTM_cell_bw, 
            output_keep_prob = dropout_keep_prob
        )

        (RNNout_fw, RNNout_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                    cell_fw = LSTM_drop_fw,
                                    cell_bw = LSTM_drop_bw,
                                    inputs = input_data,
                                    initial_state_fw=LSTM_cell_fw.zero_state(BATCH_SIZE, dtype=tf.float32),
                                    initial_state_bw=LSTM_cell_bw.zero_state(BATCH_SIZE, dtype=tf.float32),
                                    parallel_iterations = 64
                                )

    lastoutput = tf.concat(values = [RNNout_fw[:, -1, :], RNNout_bw[:, -1, :]], axis = 1)

    with tf.name_scope("FC"):
        # pred = tf.layers.batch_normalization(lastoutput, axis=1, training = training)
        pred = tf.layers.batch_normalization(lastoutput, training = training)
        pred = tf.layers.dense(pred, 128, activation = tf.nn.relu)
        pred = tf.nn.dropout(pred, dropout_keep_prob)

        # pred = tf.layers.batch_normalization(pred, axis=1, training = training)
        pred = tf.layers.batch_normalization(pred, training = training)
        pred = tf.layers.dense(pred, 128, activation = tf.nn.relu)
        pred = tf.nn.dropout(pred, dropout_keep_prob)

        # pred = tf.layers.batch_normalization(pred, axis=1, training = training)
        pred = tf.layers.batch_normalization(pred, training = training)
        pred = tf.layers.dense(pred, 128, activation = tf.nn.relu)
        pred = tf.nn.dropout(pred, dropout_keep_prob)

        # pred = tf.layers.batch_normalization(pred, axis=1, training = training)
        pred = tf.layers.batch_normalization(pred, training = training)
        pred = tf.layers.dense(pred, 64, activation = tf.nn.relu)
        pred = tf.layers.dropout(pred, rate = dropout_keep_prob)

        # pred = tf.layers.batch_normalization(pred, axis=1, training = training)
        pred = tf.layers.batch_normalization(pred, training = training)
        pred = tf.layers.dense(pred, 2, activation = tf.nn.softmax)

    with tf.name_scope("CrossEntropy"):
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits = pred, 
                    labels = labels
                )
        loss = tf.reduce_mean(cross_entropy, name = "loss")

    with tf.name_scope("Accuracy"):
        Accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(pred, 1), 
                        tf.argmax(labels, 1)
                    ), 
                    dtype = tf.float32
                ),
                name = "accuracy"
            )

    with tf.name_scope("Optimizer"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        optimizer = tf.group([optimizer, update_ops])

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss, training

# def define_graph():
#     """
#     Implement your model here. You will need to define placeholders, for the input and labels,
#     Note that the input is not strings of words, but the strings after the embedding lookup
#     has been applied (i.e. arrays of floats).

#     In all cases this code will be called by an unaltered runner.py. You should read this
#     file and ensure your code here is compatible.

#     Consult the assignment specification for details of which parts of the TF API are
#     permitted for use in this function.

#     You must return, in the following order, the placeholders/tensors for;
#     RETURNS: input, labels, optimizer, accuracy and loss
#     """

#     """
#     training_data_embedded[exampleNum., ]
#     input data is placeholder, size NUM_SAMPLES x MAX_WORDS_IN_REVIEW x EMBEDDING_SIZE
#     labels placeholder,
#     dropout_keep_prob placeholder,
#     optimizer is function with placeholder input_data, labels, dropout_keep_prob
#     Accuracy, loss is function with placeholder input_data, labels
#     """
#     lstm_hidden_unit = 256

#     learning_rate = 0.001

#     dropout_keep_prob = tf.placeholder_with_default(0.6, shape=(), name='drop_rate')

#     input_data = tf.placeholder(
#         tf.float32, 
#         shape=(BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE),
#         name="input_data"
#     )

#     input_data_norm = tf.layers.batch_normalization(input_data, axis=1)
#     input_data_norm = input_data

#     labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2), name="labels")

#     LSTM_cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_unit)
#     LSTM_cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_unit)

#     LSTM_drop_fw = tf.nn.rnn_cell.DropoutWrapper(
#         cell = LSTM_cell_fw, 
#         output_keep_prob = dropout_keep_prob
#     )

#     LSTM_drop_bw = tf.nn.rnn_cell.DropoutWrapper(
#         cell = LSTM_cell_bw, 
#         output_keep_prob = dropout_keep_prob
#     )

#     (RNNout_fw, RNNout_bw), _ = tf.nn.bidirectional_dynamic_rnn(
#                                 cell_fw = LSTM_drop_fw,
#                                 cell_bw = LSTM_drop_bw,
#                                 inputs = input_data_norm,
#                                 initial_state_fw=LSTM_cell_fw.zero_state(BATCH_SIZE, dtype=tf.float32),
#                                 initial_state_bw=LSTM_cell_bw.zero_state(BATCH_SIZE, dtype=tf.float32),
#                                 parallel_iterations = 16
#                             )

#     last_output = []

#     for i in range(1):
#         last_output.append(RNNout_fw[:, -i, :])
#         last_output.append(RNNout_bw[:, -i, :])

#     lastoutput = tf.concat(last_output, 1)


#     with tf.name_scope("fc_layer"):
#         lastoutput_norm = tf.layers.batch_normalization(lastoutput, axis=1)
#         # lastoutput_norm = lastoutput

#         pred = tf.layers.dense(lastoutput_norm, 128, activation = tf.nn.relu)
#         pred = tf.layers.batch_normalization(pred, axis=1)
#         pred = tf.layers.dropout(pred, rate = dropout_keep_prob)
#         pred = tf.layers.dense(pred, 128, activation = tf.nn.relu)
#         pred = tf.layers.batch_normalization(pred, axis=1)
#         pred = tf.layers.dropout(pred, rate = dropout_keep_prob)
#         pred = tf.layers.dense(pred, 128, activation = tf.nn.relu)
#         pred = tf.layers.batch_normalization(pred, axis=1)
#         pred = tf.layers.dropout(pred, rate = dropout_keep_prob)
#         pred = tf.layers.dense(pred, 64, activation = tf.nn.relu)
#         pred = tf.layers.batch_normalization(pred, axis=1)
#         pred = tf.layers.dropout(pred, rate = dropout_keep_prob)

#         pred = tf.layers.dense(pred, 2, activation = tf.nn.softmax)

#     cross_entropy = \
#         tf.nn.softmax_cross_entropy_with_logits_v2(
#                 logits = pred, 
#                 labels = labels
#             )

#     Accuracy = tf.reduce_mean(
#             tf.cast(
#                 tf.equal(
#                     tf.argmax(pred, 1), 
#                     tf.argmax(labels, 1)
#                 ), 
#                 dtype = tf.float32
#             ),
#             name = "accuracy"
#         )

#     loss = tf.reduce_mean(cross_entropy, name = "loss")
#     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#     # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

#     return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss