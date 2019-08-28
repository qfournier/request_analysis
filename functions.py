import sys
import glob
import time
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer


def load_data():
    counts, times, syscalls, states, duration = [], [], [], [], []

    for filename in glob.glob("data/*_count.csv"):

        with open(filename, "r") as f_counts:
            for record in f_counts:
                counts.append(list(map(float, record.split(",")[1:-1])))
                duration.append(float(record.split(",")[-1]))

        with open(filename.replace("count", "times"), "r") as f_times:
            for record in f_times:
                times.append(list(map(float, record.split(",")[1:-1])))

        with open(filename.replace("count", "syscalls"), "r") as f_syscalls:
            for record in f_syscalls:
                syscalls.append(record.split(",")[1:-1])

        with open(filename.replace("count", "states"), "r") as f_states:
            for record in f_states:
                states.append(list(map(int, record[30:-2])))

    # check they all have the same number of samples
    assert (len(counts) == len(times) and len(counts) == len(syscalls)
            and len(counts) == len(states) and len(counts) == len(duration))

    return counts, times, syscalls, states, duration


# convert system call sequences in a bag of word (bow) fashion
def seq2bow(syscalls):
    vocab = list(set([s for sequence in syscalls for s in sequence]))
    return (vocab, [[s.count(w) for w in vocab] for s in syscalls])


def seq2tfidf(syscalls):
    vocab = list(set([s for sequence in syscalls for s in sequence]))
    df = [sum([w in s for s in syscalls]) for w in vocab]
    return (vocab,
            [[s.count(w) / d for w, d in zip(vocab, df)] for s in syscalls])


def time_statistics(X):
    X = sorted(X)
    print("\t{:20s}: {:10}\n".format("Size", len(X)))
    print("Duration:")
    print("\t{:20s}: {:10.2f} ms".format("Min", min(X)))
    print("\t{:20s}: {:10.2f} ms".format("Max", max(X)))
    print("\t{:20s}: {:10.2f} ms".format("Median", X[int(len(X) / 2)]))
    print("\t{:20s}: {:10.2f} ms".format("Mean", np.mean(X)))
    print("\t{:20s}: {:10.2f} ms\n".format("Std", np.std(X)))
    print("Probabilities")
    print("\t{:20s}: {:10.3%}".format("P(duration > 200ms)",
                                      np.mean([x > 200 for x in X])))
    print("\t{:20s}: {:10.3%}".format("P(duration > 250ms)",
                                      np.mean([x > 250 for x in X])))
    print("\t{:20s}: {:10.3%}\n".format("P(duration > 300ms)",
                                        np.mean([x > 300 for x in X])))


def syscalls_statistics(X):
    S = [len(x) for x in X]
    D = [len(set(x)) for x in X]
    S = sorted(S)
    print("System Call Sequence Length:")
    print("\t{:20s}: {:10.2f}".format("Min", min(S)))
    print("\t{:20s}: {:10.2f}".format("Max", max(S)))
    print("\t{:20s}: {:10.2f}".format("Median", S[int(len(S) / 2)]))
    print("\t{:20s}: {:10.2f}".format("Mean", np.mean(S)))
    print("\t{:20s}: {:10.2f}\n".format("Std", np.std(S)))
    print("Number of Distinct System Calls:")
    print("\t{:20s}: {:10.2f}".format("Min", min(D)))
    print("\t{:20s}: {:10.2f}".format("Max", max(D)))
    print("\t{:20s}: {:10.2f}".format("Median", D[int(len(D) / 2)]))
    print("\t{:20s}: {:10.2f}".format("Mean", np.mean(D)))
    print("\t{:20s}: {:10.2f}\n".format("Std", np.std(D)))


def extrema(X):
    X = sorted(X)
    print(*np.round(X[:5], 2), sep=", ", end="")
    print(" ... ", end="")
    print(*np.round(X[-5:], 2), sep=", ")


def corr_coeff(A, B, plot=True):
    cov = [np.corrcoef(a, b)[0, 1] for a, b in zip(A, B)]
    if plot:
        plt.figure(figsize=(6, 4), dpi=300)
        plt.hist(cov, bins=100)
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        plt.xlim((0, 1))
        plt.show()
        plt.close()
    return cov


def split_clusters(cluster, time):
    duration_per_cluster = [[] for _ in range(len(set(cluster)) + 1)]
    for c, t in zip(cluster, time):
        duration_per_cluster[c].append(t)
    # remove empty clusters (DBSCAN)
    duration_per_cluster.remove([])
    return duration_per_cluster


def distribution_clustering(duration_per_cluster, name=None):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 4), dpi=300)
    plt.subplots_adjust(wspace=0.1)
    o = len(duration_per_cluster) - 1
    for i, d in enumerate(duration_per_cluster):
        sns.kdeplot(d, gridsize=100000, bw=50, ax=ax1, color="C{}".format(i))
        ax1.hist(
            d,
            bins=int((max(d) - min(d)) / 60),
            density=True,
            color="C{}".format(i),
            alpha=0.25)
        ax1.set_xlim((0, 1950))
    for i, d in enumerate(duration_per_cluster):
        sns.kdeplot(
            d,
            gridsize=100000,
            bw=50,
            ax=ax2,
            label="Outliers ({})".format(len(d))
            if i == o else "Cluster {} ({})".format(i + 1, len(d)),
            color="C{}".format(i))
        ax2.hist(
            d,
            bins=int((max(d) - min(d)) / 60),
            density=True,
            color="C{}".format(i),
            alpha=0.25)
        ax2.set_xlim((21050, 23000))
        ax2.legend()

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright=False)
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', which='both', right=False)

    # break lines
    d = .01
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    ax1.set_xlabel('Duration (ms)')
    ax1.xaxis.set_label_coords(1.2, -0.1)
    ax1.set_ylabel('Probability')
    plt.legend(frameon=False)
    if name is not None:
        plt.savefig("figures/{}_dist".format(name))
    plt.show()
    plt.close()


def statistics_clustering(duration_per_cluster, syscalls_per_cluster=None):
    for i in range(len(duration_per_cluster)):
        print("-" * 50)
        print("Cluster {}".format(i))
        print("-" * 50)
        time_statistics(duration_per_cluster[i])
        if syscalls_per_cluster:
            syscalls_statistics(syscalls_per_cluster[i])


def visualization_cluster(data,
                          cluster,
                          duration,
                          method,
                          limit=None,
                          name=None):
    data = np.asarray(data)
    if limit:
        print("Randomly sample {} values".format(limit))
        data, _, cluster, _, duration, _ = train_test_split(
            data,
            cluster,
            duration,
            train_size=limit,
            test_size=len(data) - limit,
            stratify=cluster)
    projection = method.fit_transform(data)
    # rescale x and y between 1 and 1000
    x = (projection[:, 0] - min(projection[:, 0])) / (
        max(projection[:, 0]) - min(projection[:, 0]))
    x = x * 999 + 1
    y = (projection[:, 1] - min(projection[:, 1])) / (
        max(projection[:, 1]) - min(projection[:, 1]))
    y = y * 999 + 1

    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(x, y, s=2, c=["C0" if c == -1 else "C1" for c in cluster])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim((0.5, 1500))
    plt.ylim((0.5, 1500))
    legend_elements = [
        Patch(facecolor='C0', label='Outlier'),
        Patch(facecolor='C1', label='Normal')
    ]
    plt.legend(frameon=False, handles=legend_elements)
    plt.tight_layout()
    if name is not None:
        plt.savefig("figures/{}_cluster_viz".format(name))
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(
        x,
        y,
        s=2,
        c=duration,
        cmap=mpl.cm.get_cmap('plasma_r'),
        vmin=0,
        vmax=400)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim((0.5, 1500))
    plt.ylim((0.5, 1500))
    plt.colorbar(extend='max')
    plt.tight_layout()
    if name is not None:
        plt.savefig("figures/{}_duration_viz".format(name))
    plt.show()
    plt.close()


def cluster_outliers(states,
                     cluster,
                     range_k,
                     normalize=False,
                     log=False,
                     name="kmeans_outliers"):

    vocab, states_bow = seq2bow(states)

    if normalize:
        scaler = StandardScaler()
        states_bow_norm = scaler.fit_transform(states_bow)
        states_bow_norm_outlier = [
            v for c, v in zip(cluster, states_bow_norm) if c == -1
        ]

    # get the outliers
    states_bow_outlier = [v for c, v in zip(cluster, states_bow) if c == -1]
    states_bow_normal = [v for c, v in zip(cluster, states_bow) if c != -1]

    tranlation = {
        0: "UNKNOWN",
        1: "SYSTEMCALL",
        2: "USERMODE",
        3: "BLOCKED_IO",
        4: "BLOCKED_CPU",
        5: "BLOCKED_WAITKERNEL",
        7: "BLOCKED_WAITPROCESS",
        6: "BLOCKED_FOR_FUTEX",
        8: "BLOCKED_CPU_FUTEX",
        9: "TRASH"
    }

    vocab = [tranlation[v] for v in vocab]
    inertia = []

    for k in range_k:
        kmeans = KMeans(
            n_clusters=k, max_iter=999, n_init=20, random_state=42, n_jobs=-1)
        if normalize:
            cluster_outlier = kmeans.fit_predict(states_bow_norm_outlier)
        else:
            cluster_outlier = kmeans.fit_predict(states_bow_outlier)
        # get inertia
        inertia.append(kmeans.inertia_)
        # plot feature count i each cluster
        states_counts_per_cluster = split_clusters(cluster_outlier,
                                                   states_bow_outlier)
        try:
            plt.figure(figsize=(8, 4), dpi=300)
            legend_elements = []

            for i, state_cluster in enumerate(
                [states_bow_normal] + states_counts_per_cluster, 0):
                for j, s in enumerate(list(zip(*state_cluster))):
                    plt.barh(
                        (k + 2) * j + i,
                        np.mean(s),
                        xerr=np.std(s),
                        color="C{}".format(i - 1) if i > 0 else "gray",
                        error_kw={"elinewidth": 1},
                        log=log)

                legend_elements.append(
                    Patch(
                        facecolor='C{}'.format(i - 1) if i > 0 else "gray",
                        label='Outlier {} ({})'.format(i, len(state_cluster))
                        if i > 0 else 'Normal  ({})'.format(
                            len(state_cluster))))
            plt.yticks(np.arange(k / 2, (k + 2) * len(vocab), (k + 2)), vocab)
            plt.xlabel("Counts")
            if log:
                plt.xlim((10, 3e3))
            plt.legend(frameon=False, handles=legend_elements)
            plt.tight_layout()
            plt.savefig("figures/{}_{}".format(name, k))
            plt.show()
            plt.close()
        except ValueError:
            print(
                "Cannot plot with log scale: min() arg is an empty sequence",
                file=sys.stderr)
        finally:
            plt.close("all")

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(range_k, inertia)
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig("figures/{}_inertia".format(name))
    plt.show()
    plt.close()


def pipeline(data,
             duration,
             model,
             bow=False,
             tfidf=False,
             viz_model=None,
             limit=1000,
             name=None):

    if bow:
        x = seq2bow(data)[1]
    elif tfidf:
        x = seq2tfidf(data)[1]
    else:
        x = data

    cluster = model.fit_predict(x)
    duration_per_cluster = split_clusters(cluster, duration)
    statistics_clustering(duration_per_cluster)
    distribution_clustering(duration_per_cluster, name)
    if viz_model:
        visualization_cluster(x, cluster, duration, viz_model, limit, name)
    return cluster


def perf(syscalls, states, times, duration, model, n=10):
    time_preprocessing = []
    time_outliers = []
    time_clustering = []
    time_ngram = []

    for i in range(n):
        # features bag of word
        start = time.time()
        states_bow = seq2bow(states)[1]
        time_preprocessing.append(time.time() - start)

        # outlier detection based on feature duration
        start = time.time()
        cluster = model.fit_predict(times)
        time_outliers.append(time.time() - start)

        # normalizing (not included)
        scaler = StandardScaler()
        states_bow_norm = scaler.fit_transform(states_bow)
        states_bow_norm_outlier = [
            v for c, v in zip(cluster, states_bow_norm) if c == -1
        ]

        # clustering
        start = time.time()
        kmeans = KMeans(
            n_clusters=3, max_iter=999, n_init=20, random_state=42, n_jobs=-1)
        cluster_outlier = kmeans.fit_predict(states_bow_norm_outlier)
        time_clustering.append(time.time() - start)

        # whole ngram analysis
        start = time.time()
        ngram_statistics(
            syscalls, cluster, cluster_outlier, ngram_range=(1, 3))
        time_ngram.append(time.time() - start)

    print("Preprocessing: {:.3f} ± {:.3f}s ".format(
        np.mean(time_preprocessing), np.std(time_preprocessing)))
    print("Outlier detection: {:.3f} ± {:.3f}s ".format(
        np.mean(time_outliers), np.std(time_outliers)))
    print("Clustering of outliers: {:.3f} ± {:.3f}s ".format(
        np.mean(time_clustering), np.std(time_clustering)))
    print("Ngram analysis: {:.3f} ± {:.3f}s ".format(
        np.mean(time_ngram), np.std(time_ngram)))



def get_ngram(data, ngram_range):
    data = [" ".join(s) for s in data]
    vectorizer = CountVectorizer(ngram_range=ngram_range, lowercase=False)
    count_per_record = vectorizer.fit_transform(data)
    mean_count, sem_count = [], []
    for i in range(count_per_record.shape[1]):
        col = count_per_record.getcol(i).toarray()
        mean_count.append(np.mean(col))
        sem_count.append(stats.sem(col))
    return mean_count, sem_count, vectorizer.get_feature_names()


def ngram_statistics(syscalls,
                     cluster,
                     cluster_outlier,
                     ngram_range=[2, 3],
                     verbose=False,
                     csv_file=None):
    # get number of cluster
    n_cluster = len(set(cluster_outlier))

    # split normal and outliers
    syscalls_normal = [v for c, v in zip(cluster, syscalls) if c != -1]
    syscalls_outlier = [v for c, v in zip(cluster, syscalls) if c == -1]

    # split outliers in clusters
    syscalls_cluster = split_clusters(cluster_outlier, syscalls_outlier)

    # n-gram
    normal_count, normal_sem, normal_ngram = get_ngram(syscalls_normal,
                                                       ngram_range)

    outlier_counts, outlier_sems, outlier_ngrams = [], [], []
    for s in syscalls_cluster:
        count, sem, ngram = get_ngram(s, ngram_range)
        outlier_counts.append(count)
        outlier_sems.append(sem)
        outlier_ngrams.append(ngram)

    # vocabulary
    ngram = set([o for outlier_ngram in outlier_ngrams
                 for o in outlier_ngram] + normal_ngram)

    # create dataframe
    df = pd.DataFrame(ngram, columns=['Ngram'])

    df["Normal mean"] = [
        normal_count[normal_ngram.index(word)] if word in normal_ngram else 0.0
        for word in ngram
    ]

    df["Normal sem"] = [
        normal_sem[normal_ngram.index(word)] if word in normal_ngram else 0.0
        for word in ngram
    ]

    for i in range(n_cluster):
        df["Cluster {} mean".format(i)] = [
            outlier_counts[i][outlier_ngrams[i].index(word)]
            if word in outlier_ngrams[i] else 0.0 for word in ngram
        ]

        df["Cluster {} sem".format(i)] = [
            outlier_sems[i][outlier_ngrams[i].index(word)]
            if word in outlier_ngrams[i] else 0.0 for word in ngram
        ]

    df = df.sort_values(by="Ngram")

    if verbose:
        print(df)
    if csv_file is not None:
        df.to_csv(csv_file)