import tensorflow as tf
import numpy as np
from fastText import fastTextB
import sys
import os
sys.path.insert(1, 'C:\\Users\\puria\\source\\repos\\puria-radmard\\vectorai\\')
from word_embeddings import embed_and_augment_data as em_aug

class fastTextPredictor:
    def __init__(self):
        
        self.dictionary = {0 : "Company", 1: "Date", 2: "Location", 3: "Vessel"}
        self.graph = tf.Graph()


    def classify(self, entry):
        "Takes entry as an iterable type, not DataFrame"

        with self.graph.as_default():
            with tf.Session() as sess:
                self.metasaver = tf.train.import_meta_graph("fastText_model/vectorai_fastText_classifier.ckpt.meta")
                self.metasaver.restore(sess, tf.train.latest_checkpoint(os.path.dirname("fastText_model/vectorai_fastText_classifier.ckpt.meta")))
                log_op = self.graph.get_operation_by_name("logits")
                
                x_in = [em_aug(e) for e in entry]

                pred_logs = sess.run(["logits:0"], feed_dict={"sentence:0": x_in})[0]
            
        
        preds = np.argmax(pred_logs, axis = 1)
        return preds

from compiling_data import X_test, y_test

dictionary = {"Company": 0, "Date": 1, "Location": 2, "Vessel": 3}

preds = fastTextPredictor().classify(X_test)
reals = [dictionary[a] for a in y_test["type"]]

acc = np.mean(preds == reals)
print(acc)