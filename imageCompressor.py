#  Author aashish-ak

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from scipy import misc
print('Image Compressor ML Algorithm')
arr = misc.imread('dp.jpg')
img_size = arr.shape
arr_ = arr.astype(float)
arr_ = np.divide(arr, 255)
arr_ = arr_.reshape(img_size[0]*img_size[1],3)
kmeans = KMeans(n_clusters = 20)
kmeans.fit(arr_)
arr_compressed = kmeans.cluster_centers_[kmeans.labels_, :]
arr_compressed = arr_compressed.reshape(img_size[0], img_size[1], 3)
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
plt.imshow(arr)
a.set_title('Original')
a = fig.add_subplot(1, 2, 2)
plt.imshow(arr_compressed)
a.set_title('Compressed to 20 colors')
plt.show()
