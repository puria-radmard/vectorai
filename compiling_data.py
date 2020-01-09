import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


dfcom = pd.read_csv("Latest Data/companies.csv").set_index("index")
dfdat = pd.read_csv("Latest Data/dates.csv").set_index("index")
dfloc = pd.read_csv("Latest Data/locations.csv").set_index("index")
dfves = pd.read_csv("Latest Data/vessels.csv").set_index("index")

df = pd.concat([dfcom, dfdat, dfloc, dfves], axis = 0)#.reset_index()

usable_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '&', "'", '(', ')', ',', '-', '.', '/', ':', '[', ']']#, ' ']

def splitwords(word):
    word = str(word)
    return[x for x in word if x in usable_char]

df["data"] = df["data"].apply(splitwords)

X_train, X_test, y_train, y_test = train_test_split(df["data"], df.drop("data", axis = 1))    # For local run


# For azure run
from word_embeddings import embed_and_augment_data as em_aug
df["data"] = df["data"].apply(em_aug)
X_train_, X_test_, y_train_, y_test_ = train_test_split(df["data"], df.drop("data", axis = 1))

np.save("splitted_data/X_train", X_train_, allow_pickle=True)
np.save("splitted_data/y_train", y_train_, allow_pickle=True)
np.save("splitted_data/X_test", X_test_, allow_pickle=True)
np.save("splitted_data/y_test", y_test_, allow_pickle=True)