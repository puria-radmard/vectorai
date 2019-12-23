import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dfcom = pd.read_csv("Latest Data/companies.csv").set_index("index")
dfdat = pd.read_csv("Latest Data/dates.csv").set_index("index")
dfloc = pd.read_csv("Latest Data/locations.csv").set_index("index")
dfves = pd.read_csv("Latest Data/vessels.csv").set_index("index")

df = pd.concat([dfcom, dfdat, dfloc, dfves], axis = 0)#.reset_index()

# Now we need to OHE this big DF, with first one removed to stop colinearity
# dfencoded = pd.get_dummies(df, prefix=['type'], columns=['type'], drop_first = True)

# X_train, X_test, y_train, y_test = train_test_split(dfencoded["data"], dfencoded.drop("data", axis = 1))
X_train, X_test, y_train, y_test = train_test_split(df["data"], df.drop("data", axis = 1))