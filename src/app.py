import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')

df = df_raw[['MedInc','Latitude','Longitude']]

escalador=StandardScaler()
df_norm=escalador.fit_transform(df)

rango_n_clusters = [2, 3, 4, 5, 6, 7, 8,9,10]
silhouette_avg = []
for num_clusters in rango_n_clusters:
# fit Kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df_norm)
    cluster_labels = kmeans.labels_
# calcular silhouette
    silhouette_avg.append(silhouette_score(df_norm, cluster_labels))

kmeans = KMeans(n_clusters=2)
kmeans.fit(df_norm)

# hago la prediccion 
prediccion = kmeans.predict(df_norm)

df2=escalador.inverse_transform(df_norm)

df2=pd.DataFrame(df2,columns=['MedInc','Latitude','Longitude'])

df2['Cluster'] = kmeans.labels_

# guardo el dataset nuevo
df2.to_csv('../data/processed/df_final.csv', index = False)