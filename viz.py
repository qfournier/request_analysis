from functions import visualization_cluster
from functions import load_data
from sklearn.cluster import DBSCAN
from sklearn.manifold import Isomap

# load data
_, times, _, _, duration = load_data()

# convert feature times and duration into ms
duration = [d / 1e6 for d in duration]
times = [[t / 1e6 for t in sample] for sample in times]

# remove 0.14 and 4.22 are most likely to be measurment errors
for _ in range(2):
    index = duration.index(min(duration))
    del times[index]
    del duration[index]

model = DBSCAN(eps=25, min_samples=100, n_jobs=-1, algorithm='ball_tree')

# visualization with Isomap
viz_model = Isomap(n_neighbors=15, n_components=2, n_jobs=-1)
cluster = model.fit_predict(times)
visualization_cluster(times, cluster, duration, viz_model, 25000,
                      "features_duration_isomap")