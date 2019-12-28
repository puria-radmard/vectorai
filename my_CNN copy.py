# All credits for CNN architecture go to github/brightmart. Original code can be found https://github.com/brightmart/text_classification/blob/master/a02_TextCNN/p7_TextCNN_model.py
# I have removed annotations and added some intermediary layers.

# A CNN was initially chosen over any form of recurrent network due to the length of text clasffied - no need to understand
# context or sentiment during the classification, just the spatial spread of different characters in the entry.

# However, the CNN's processing layer requires the embedding of the text input to process as vectoral data. Because of our context, word
# embedding is not an option, so I have tried to devise a character embedding system, also based on these articles:
# https://blogs.oracle.com/datascience/character-based-neural-networks-for-nlp-in-the-real-world
# https://arxiv.org/pdf/1502.01710v5.pdf
# https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10      
# (the second article uses an RCNN, which again was decided against)

import tensorflow as tf
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=Warning)

class TextCNN:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,
                is_training = False,initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,decay_rate_big=0.50, leaky_alpha= 0.2, l2_lambda=0.0001):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size                      # For us this is number of characters involved
        self.embed_size=embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes=filter_sizes # it is a list of int. e.g. [3,4,5]
        self.num_filters=num_filters
        self.initializer=initializer
        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.
        self.clip_gradients = clip_gradients
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")
        self.leaky_alpha = leaky_alpha
        self.l2_lambda = l2_lambda

        with tf.name_scope("placeholders"):
            self.X_in = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name='X_in') # (batch size, input length, in_channels)
            self.y_in = tf.placeholder(tf.int32, [self.batch_size,], name = 'y_in')    # No multiclassification needed for us
            self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob") # May later change to a new one for each layer
            self.iter = tf.placeholder(tf.int32) #training iteration
            self.tst=tf.placeholder(tf.bool)                                # We will be using a multiple layer CNN, so no need to check like brightmart does

        with tf.name_scope("training"):
            self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
            self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
            self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
            self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
            self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
            self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.logits = self.inference_and_weight_instantiation()
        self.possibility = tf.nn.sigmoid(self.logits)

        self.loss_val = self.loss()
        self.train_op = self.train()

        self.predictions = tf.argmax(self.logits, 1, name='predictions')
        print(self.predictions)
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.y_in)                # Boolean same or no
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")    # 0 - 1

        # I first tried to work with conv1d, but realised this would only allow one filter at a time, 
        
    def inference_and_weight_instantiation(self):

        # I did way too much research into character embedding before realising characters are also included in standard
        # word embedding libraries, so splitting the inputs up with spaces is the equivalent

        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size])                                 # ,initializer=self.initializer)        #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes])                    #,initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size] #ADD 2017.06.09

        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.X_in)#[None,sentence_length,embed_size]
        
        # Had to change up this line to avoid rank compatability issues
        self.embeddings_expanded=tf.expand_dims(self.embedded_words,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        print(tf.shape(self.embeddings_expanded))
        #self.embedded_transpose = tf.reshape(self.embedded_words, shape = [tf.shape(self.embedded_words)[0]] + [self.sequence_length, self.embed_size, 1])

        h = self.cnn_layers()
        
        with tf.name_scope('output'):
            logits = tf.matmul(h, self.W_projection) + self.b_projection
        return logits

    def cnn_layers(self):

        # Goes through each 1d filter size for text convolution, producing a series of layers with num_filters for each size
        pooled_outputs = []                     
        for i, filter_size in enumerate(self.filter_sizes):
            
            with tf.variable_scope("conv-pooling-{}".format(filter_size)):
                
                # First layer: CNN layer -> batch normalisation (for training time too) -> leaky relu activation
                # Produce a learnable conv2d filter of size (filter height - number of chars, filter width - size of embedded vectors, in channels - 1 for text, out channels - number of filters to further convolve text)
                filter = tf.get_variable("filter-size-{}".format(filter_size), [filter_size, self.embed_size, 1, self.num_filters])
    
                conv1 = tf.nn.conv2d(self.embeddings_expanded, filter, strides = [1,1,1,1], padding = 'SAME', name = 'conv1')
                conv1 = tf.contrib.layers.batch_norm(conv1, is_training = self.is_training_flag, scope = 'cnn-layer1')
                print(i, "conv1: ", conv1)
                # Learned bias for this layer
                b1 = tf.get_variable("b-{}".format(filter_size), [self.num_filters])
                # Made this leaky to prevent neuron death
                h = tf.nn.leaky_relu(tf.nn.bias_add(conv1, b1), alpha = self.leaky_alpha, name = "relu")
                print("h out of relu: ", h)
                
                #h = tf.reshape(h, [-1, self.sequence_length, self.num_filters, 1])
                #print("h into conv2: ", h)

                # Second layer: Same again but with no leaky relu, just relu
                filter2 = tf.get_variable("filter2-%s" % filter_size,[filter_size, self.num_filters, 128, self.num_filters])   #,initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1,1,1,1], padding="SAME",name="conv2")  # shape:[batch_size, convolved length: sequence_length-filter_size*2+2, final convolved vector: 1, num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                # Had to change
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length,1,num_filters]
                #h = tf.reshape(h, [-1, self.sequence_length, self.num_filters, 1])
                #print("Then after resize: {}".format(tf.shape(h)))

                # Third layer: Max-pooling, changed somewhat from source
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1,self.sequence_length, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                pooling_avg=tf.squeeze(tf.reduce_mean(pooling_max,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_avg)
                #pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_avg)  # h:[batch_size,sequence_length,1,num_filters]
                print("pooled_outputs from size {}: ".format(filter_size), pooled_outputs)

        h = tf.concat(pooled_outputs, axis = 1)      # Of shape (batch size, num_filters * number of different filter sizes)
        #h = tf.reshape(pooled_outputs, [tf.shape(h)[0]/self.num_filters_total, self.num_filters_total])

        print("Final convolution output before dropout: ", tf.shape(h))

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob = self.dropout_keep_prob)     # Again, may change this so that there's one on each layer
        return h

    def loss(self):   
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y_in, logits = self.logits)
            # Didn't think of this, will try to model for different parameters also
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss+l2_losses
        return loss
    
    def train(self):
        # To prevent momentum overstep
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        # Adam optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

import string

def characterise_inputs(word):
    return " ".join([x if x in string.printable else " " for x in str(word)])

from compiling_data import X_train, y_train, X_test, y_test

X_train = X_train.apply(characterise_inputs)
X_test = X_train.apply(characterise_inputs)

def test():
    #below is a function test; if you use this for text classifiction, you need to transform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=4
    learning_rate=0.001
    batch_size=5
    decay_steps=1000
    decay_rate=0.95
    sequence_length=5                 ### WILL NEED TWEAKING
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1.0
    filter_sizes=[2,3,4]
    num_filters=128
    print("wow")
    textRNN=TextCNN(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
    print("wow2")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            print("WOW3")
            input_x=np.random.randn(batch_size,sequence_length) #[None, self.sequence_length]
            input_x[input_x>=0]=1
            input_x[input_x <0] = 0
            input_y = compute_single_label(input_x)
            #print(input_x, input_y)
            loss,possibility,W_projection_value,_=sess.run([textRNN.loss_val,textRNN.possibility,textRNN.W_projection,textRNN.train_op],
                                                    feed_dict={textRNN.X_in:input_x,textRNN.y_in:input_y,
                                                               textRNN.dropout_keep_prob:dropout_keep_prob,textRNN.tst:False,
                                                               textRNN.is_training_flag:is_training})                                                              # Had to be added
            print(i,"loss:",loss,"-------------------------------------------------------")
            print("label:",input_y)#print("possibility:",possibility)

def compute_single_label(listt):
    outlist = []
    for l in listt:
        outlist.append(sum(l)%4)
    return outlist

test()