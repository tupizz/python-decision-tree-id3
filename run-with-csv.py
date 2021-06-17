import pandas as pd
import numpy as np
from id3 import DecisionTreeClassifier
import subprocess

df = pd.read_csv("kr-vs-kp.csv")

print(df.head())

LABEL_NAME = 'class'

# passa tudo menos o label (ou attr clase)
X = np.array(df.drop(LABEL_NAME, axis=1).copy())

# os labels
Y = np.array(df[LABEL_NAME].copy())

# pega o nome das colunas
FEATURE_NAMES = list(df.keys())[:-1]

tree_clf = DecisionTreeClassifier(X=X, feature_names=FEATURE_NAMES, labels=Y)
print("Entropia da tabela inicial {:.4f}".format(tree_clf.entropy))

# run algorithm id3 to build a tree
tree_clf.id3()

# copia o resultado para o ctrl c para colar no site:
subprocess.run("pbcopy", universal_newlines=True, input=tree_clf.printTree())