from my_CNN import TextCNN
from word_embeddings import embed_and_augment_data as em_aug
from compiling_data import X_train, X_test, y_train, y_test
import numpy as np
import tensorflow as tf

dictionary = {"Company": 0, "Date": 1, "Location": 2, "Vessel": 3}
num_classes=4
learning_rate=0.001
batch_size=5
decay_steps=1000
decay_rate=0.95
sequence_length=40
embed_size=100
is_training=True
dropout_keep_prob=1.0
filter_sizes=[4,5,6]
num_filters=128

textRNN=TextCNN(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,embed_size,is_training)

n_epochs = 1
saver = tf.train.Saver()

def train_and_save():
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        for n in range(n_epochs):

            for i in range(len(X_train) % batch_size):

                input_x = np.stack(X_train.iloc[i*batch_size : batch_size*(i+1)].apply(em_aug).values)         # New augmentation every time
                input_y = y_train.iloc[i*batch_size : batch_size*(i+1)].values
                input_y = np.vectorize(dictionary.get)(input_y).reshape(-1)
                loss,possibility,W_projection_value,_=sess.run([textRNN.loss_val,textRNN.possibility,textRNN.W_projection,textRNN.train_op],
                                                        feed_dict={textRNN.X_in:input_x,textRNN.y_in:input_y,
                                                                    textRNN.dropout_keep_prob:dropout_keep_prob,textRNN.tst:False,
                                                                    textRNN.is_training_flag:is_training})

            print("Batch {} of epoch {} completed, loss = {}".format(i, n, loss))
        
        save_patch = saver.save(sess, "./vectorai_word_classifier.ckpt")

train_and_save()