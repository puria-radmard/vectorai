from training_data import df, usable_char
from gensim.models import Word2Vec
import gensim

def w2vmodel(build = False, show = False, model_rev = False):

    if build:
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
    
#model(show = True)

def augment_data(word, length):
    
    # Changes data randomly (using characters from the data) to both standardise string length, and increase sample size
    