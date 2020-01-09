import sys
sys.path.insert(1, 'C:\\Users\\puria\\source\\repos\\puria-radmard\\vectorai\\')

from fastText import fastTextB
from word_embeddings import embed_and_augment_data as em_aug
from compiling_data import X_train, X_test, y_train, y_test
import pandas as pd

print(y_test.head, X_test.head)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dictionary = {"Company": [1, 0, 0, 0], "Date": [0, 1, 0, 0], "Location": [0, 0, 1, 0], "Vessel": [0, 0, 0, 1]}
learning_rate = 0.001
batch_size = 50
decay_steps = 128
decay_rate = 0.9
num_sampled = 10
sentence_len = 40
embed_size = 100
is_training = True
num_classes = 4

fast_text = fastTextB(learning_rate, batch_size, decay_steps, decay_rate,num_sampled,sentence_len,embed_size,is_training, num_classes)

n_epochs = 100
saver = tf.train.Saver()

sys.path.insert(1, 'C:\\Users\\puria\\source\\repos\\puria-radmard\\vectorai\\fastText_model')

def train_and_save():
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        X_test_in = np.stack(X_test.apply(em_aug).values)
        y_test_in = y_test.values.reshape(-1)
        y_test_in = [dictionary[y] for y in y_test_in]

        accs = []

        for n in range(n_epochs):

            for i in range(len(X_train) % batch_size):

                input_x = np.stack(X_train.iloc[i*batch_size : batch_size*(i+1)].apply(em_aug).values)         # New augmentation every time
                input_y = y_train.iloc[i*batch_size : batch_size*(i+1)].values.reshape(-1)
                input_y = [dictionary[y] for y in input_y]
                #input_x, input_y = remove_nones(input_x, input_y)           # Changed it now so that empty entries only have an empty array

                curr_eval_loss,logit,_ = sess.run([fast_text.loss_val,fast_text.logits,fast_text.train_op], #curr_eval_acc-->fast_text.accuracy
                                          feed_dict={fast_text.sentence: input_x, fast_text.labels: input_y})                                                            # Had to be added
                
                tf.get_variable_scope().reuse_variables()


            pred_logs = fast_text.logits.eval(feed_dict={fast_text.sentence: X_test_in, fast_text.labels: y_test_in})
            preds = np.argmax(pred_logs, axis = 1)
            actual = np.argmax(y_test_in, axis = 1)
            test_accuracy = np.mean(preds == actual)
            accs.append(test_accuracy)

            print("Epoch {} completed, loss = {}, test accuracy = {}".format(n, curr_eval_loss, test_accuracy))
            
        plt.plot(accs)
        plt.show()

        save_patch = saver.save(sess, "./fastText_model/vectorai_fastText_classifier.ckpt")

def remove_nones(X, y):
    inds = X[X[0].notnull()].index
    X_out = X.iloc[inds]
    y_out = y[inds]
    return X_out, y_out

train_and_save()