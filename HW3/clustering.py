import pandas as pd
from data import heights, weights, age
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

data_points = []
for i, (h, w, a) in enumerate(zip(heights, weights, age)):
    data_points.append([h, w, a])

df = pd.DataFrame(data_points, columns=['height', 'weight', 'age'])

plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend_complete = shc.dendrogram(shc.linkage(df, method='complete'))
plt.savefig('dendrogram_complete.png')

plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend_single = shc.dendrogram(shc.linkage(df, method='single'))
plt.savefig('dendrogram_single.png')
