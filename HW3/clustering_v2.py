import pandas as pd
from data import heights, weights, age
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

data_points = []
for i, (h, w, a) in enumerate(zip(heights, weights, age)):
    data_points.append([h, w, a])

df = pd.DataFrame(data_points, columns=['height', 'weight', 'age'])

similarity_matrix = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
# print(similarity_matrix)
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(df, method='complete'))
print(dend)
plt.savefig('dendrogram_complete.png')