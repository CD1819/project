import pandas as pd
import numpy as np
# !pip install graphviz
from sklearn import tree
from sklearn import datasets
  
# import some data to play with
iris = datasets.load_iris()

dataset = iris.data
df = pd.DataFrame(dataset, columns=iris.feature_names)
df['species'] = pd.Series([iris.target_names[cat] for cat in iris.target]).astype("category", categories=iris.target_names, ordered=True)

# define feature space
features = iris.feature_names
target = 'species'

HOW_DEEP_TREES = 1

clf = tree.DecisionTreeClassifier(random_state=0, max_depth=HOW_DEEP_TREES)
clf = clf.fit(df[features], df[target])

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,  feature_names=features,  
    class_names=iris.target_names,  filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data)  
graph.render('dtree_render',view=True)