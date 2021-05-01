from math import sqrt

import numpy as np
from data import heights, weights, age

data_points = []
clusters = {}
for i, (h, w, a) in enumerate(zip(heights, weights, age)):
    data_points.append([h, w, a])
    clusters[i] = [h, w, a]

similarity_matrix = []


def dist(a, b):
    d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


D = {}

for cluster1, cords1 in clusters.items():
    D[cluster1] = {}
    for cluster2, cords2 in clusters.items():
        D[cluster1][cluster2] = dist(cords1, cords2)

# minimum = (-1, float("inf"))
# tree_list = []
# element_to_merge = -1
# while len(D) > 1:
#     for cluster1, v in D.items():
#         tmp_min = list(dict(sorted(v.items(), key=lambda item: item[1])).items())[1]
#         if tmp_min[1] < minimum[1]:
#             element_to_merge = tmp_min[0]
#             minimum = ((cluster1, element_to_merge), tmp_min[1])
#     tree_list.append(minimum)
#     for point in list(minimum[0]):
#         D[minimum[0]] = []
#         del D[point]
#     # # print("Smallest distance: {}".format(min()))
#     # for cluster2, d in v.items():
#     #     print(cluster1, cluster2, d)