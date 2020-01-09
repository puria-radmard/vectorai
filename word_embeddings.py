from gensim.models import Word2Vec
import gensim
import random
import numpy as np

def w2vmodel(build = False, show = False, model_rev = False):

    if build:
        from compiling_data import df, usable_char, splitwords
        # Produce list of lists for gensim to CHARACTER embed
        embed_size = 100
        sent = [x for x in df["data"]]
        model = Word2Vec(sent, min_count = 1, size = embed_size, workers = 3, window = 3, sg = 1) # sg since smaller
        model.wv.save_word2vec_format('embed_model.bin', binary = True)
    
    else:
        model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format("embed_model.bin", binary = True, unicode_errors='ignore')

    if show:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import decomposition
        import string

        pca = decomposition.PCA(n_components = 2)
        checks = usable_char
        X_full = [model[a] for a in checks]
        X = np.array(pca.fit_transform(X_full))
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1]) 
        for i in range(len(checks)):
            ax.annotate(checks[i], (X[i, 0], X[i, 1]))

        plt.show()

# What I used to get usable_char in training_data.py
    if model_rev:
        a = []
        import string
        for s in string.printable:
            try:
                model[s]
                a.append(s)
            except:
                pass
        print(a)
    
    return model
    
model = w2vmodel(build = False, show = False, model_rev = False)

def embed_and_augment_data(word, model = model, length = 40):   # Input has to be a split word, see training_data.py

    # Changes data randomly (using characters from the data) to both standardise string length, and increase sample size
    # This was taken from this article https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28
    usable_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '&', "'", '(', ')', ',', '-', '.', '/', ':', '[', ']']#, ' ']

    outword = np.array([model[a] for a in word if a in usable_char])    # For not processed words

    while len(outword) != length:
        lng = len(outword)

        if lng == 0:
            return np.zeros((length, 100))
        
        if lng > length:                                   # Average two random adjacent vectors until length matches
            ind = np.random.randint(0, lng - 1)
            outword[ind] = 0.5*(outword[ind] + outword[ind+1])
            outword = np.delete(outword, ind+1, 0)

        if lng < length:                                   
            cointoss = random.random()
            
            if cointoss >= 0.5:                                     # Repeats random sequences of size 2,3,4 (or up to entry length)
                seq_len = np.random.randint(2, 5)
                if seq_len > lng: seq_len = lng

                ind = np.random.randint(0, lng - seq_len + 1)       # Picks a place to start for sequence replication
                seq = outword[ind:ind+seq_len]
                outword = np.insert(outword, [ind], seq, axis = 0)

            else:                                                   # Picks SECOND most similar letter to it (i.e. bypassing self)
                ind = np.random.randint(0, lng)
                sim = model.most_similar(positive=[outword[ind]], topn=2)[1][0]
                
                outword = np.insert(outword, [ind], model[sim], axis = 0)

    return outword

#aug_word = embed_and_augment_data(splitwords("Hello my name is Puria and this is the augmented version of this sentece. Enjoy!"), length = 40)

#print(aug_word.shape)
#print([model.most_similar(positive=[a], topn=2)[0][0] for a in aug_word])
#print(aug_word, "\n \n", np.shape(aug_word))