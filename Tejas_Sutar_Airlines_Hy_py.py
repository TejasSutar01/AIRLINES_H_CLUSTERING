# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:35:42 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
df=pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\COMPLETED\\CLUSTERING\\AIRLINES\\EastWestAirlines.csv")
############Normalizing the data#################
def norm_func(i):
    x=(i-i.min()/i.std())
    return (x)
df_norm=norm_func(df.iloc[:,1:])
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
help(linkage)
z=linkage(df_norm,method="complete",metric="euclidean")


plt.figure(figsize=(25,50));plt.title("Dendogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=6.,
)
plt.show()


#################NOw applying agglomerative clustering########
from sklearn.cluster import AgglomerativeClustering
h_labels=AgglomerativeClustering(n_clusters=14,affinity="euclidean",linkage="complete").fit(df_norm)
clusters_labels=pd.Series(h_labels.labels_)

df["Clusters"]=clusters_labels
df_final=df.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
df_final.to_csv("Flights.csv")
