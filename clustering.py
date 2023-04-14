from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

resultDf = pd.read_csv('clustering.csv')
clusteringDf = resultDf
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(
        clusteringDf[['positivity_rate', 'death_rate', 'mortality_rate']])

    labels = kmeans.labels_

    colors = ['red', 'green', 'blue', 'yellow', 'black', 'magenda']
    resultDf['Graph_Color'] = [colors[label] for label in labels]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(resultDf['positivity_rate'], resultDf['death_rate'],
               resultDf['mortality_rate'], c=resultDf['Graph_Color'])

    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               marker='x', s=200, linewidths=1, color='black')

    # Add labels to the plot
    ax.set_xlabel('Positivity rate')
    ax.set_ylabel('Death rate')
    ax.set_zlabel('Mortality rate')
    ax.set_title(f'{k} clusters with K-Means')

    # Show the plot
    plt.show()
