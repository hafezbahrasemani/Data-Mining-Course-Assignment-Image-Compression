import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import k_means

np.random.seed(1)


def distance_from_center(point, center):
    temp = 0.0

    for i in range(0, len(point)):
        temp += abs(point[i] - center[i]) ** 2

    return math.sqrt(temp)


def initialize_centroids(data, k):
    num_of_samples = data.shape[0]
    sample_points_ids = random.sample(range(0, num_of_samples), k)

    centroids = [tuple(data[id]) for id in sample_points_ids]
    unique_centroids = list(set(centroids))

    number_of_unique_centroids = len(unique_centroids)
    while number_of_unique_centroids < k:
        new_sample_points_ids = random.sample(range(0, num_of_samples),
                                              k - number_of_unique_centroids)
        new_centroids = [tuple(data[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))

        number_of_unique_centroids = len(unique_centroids)

    return np.array(unique_centroids)
    #
    # min_s = [data.min(axis=0), data.min(axis=1)]
    # print(min_s)
    #
    # # np.random.seed(1)
    # temp = copy.deepcopy(data)
    # np.random.shuffle(temp)
    # print(data[0], temp[0])
    # return temp[0:K]


def calculate_difference(centers, point):
    differences = np.zeros(len(centers))
    for i in range(len(centers)):
        differences[i] = np.linalg.norm(centers[i] - point)

    return differences


def specifying_clusters(data, K, max_iteration):
    centers = initialize_centroids(data, K)

    # data = np.array(data)
    data_from_cluster = np.zeros(len(data))

    iterate_num = 0
    while True:
        print("in while statement (:")
        clusters = [[] for i in range(K)]
        iterate_num += 1

        # centers_old = copy.deepcopy(centers)

        # print("Distances:--------------------------------------")
        for j in range(0, len(data)):
            distances = calculate_difference(centers, data[j])
            # print(distances)
            min_index = np.argmin(distances)
            # print(min_index)
            clusters[min_index].append(data[j])
            data_from_cluster[j] = min_index
        print("after for loop\n")

        for i in range(K):
            print("length of cluster num {} is {}".format(i, len(clusters[i])))

        for i in range(K):
            centers[i] = np.mean(clusters[i], axis=0)

        # 3- Check terminate condition
        # if terminate_condition(centers, centers_old):
        #     print("Clustering is done by difference condition"
        #             + " after " + str(iterate_num) + " iterations")
        #     break

        if iterate_num > max_iteration:
            print("Clustering is done by max_iteration condition")
            break

    return clusters, centers, data_from_cluster


def draw_data(train_data, centers, data_from_cluster, K, iterations):
    # colors = ['blue', 'green', 'yellow', 'brown', 'pink', 'magenta', 'C0', 'C2', 'C4', 'C6']
    # colors = []
    # for i in range(len(centers)):
    #     colors.append(list(np.random.choice(range(256), size=3)))
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(centers))]

    centers_x = []
    centers_y = []

    train_data_x = []
    train_data_y = []

    for i in range(len(centers)):
        centers_x.append(centers[i][0])
        centers_y.append(centers[i][1])

    for i in range(len(train_data)):
        train_data_x.append(train_data[i][0])
        train_data_y.append(train_data[i][1])

    for i in range(len(train_data)):
        plt.plot(train_data_x[i], train_data_y[i],
                 'bo', color=colors[int(data_from_cluster[i])])

    # plt.plot(train_data_x, train_data_y, 'bo', color='blue')

    # plt.plot(X_test, y_test, 'bo', color='blue')
    plt.plot(centers_x, centers_y, 's', color='red')
    plt.xlabel("K = " + str(K) + "      iterations = " + str(iterations))
    plt.show()


def draw_clustering_error(k_values, errors):
    plt.xlabel("K")
    plt.ylabel("clustering error")
    plt.plot(k_values, errors, '-bo')
    plt.show()


def compute_cluster_error(cluster, centroid):
    differences = []
    for i in range(len(cluster)):
        differences.append(distance_from_center(cluster[i], centroid))

    return np.mean(differences)


def compute_error_for_some_k(data, K, iterations):
    errors = []

    K = 0
    while K < 15:
        K += 1
        # print("---------------------The results for K = {}-----------------------\n".format(K))

        clusters, centers, data_from_cluster = \
            specifying_clusters(data, K, iterations)

        temp = []
        for i in range(len(clusters)):
            # print("The error of cluster {0} is {1}".format(i, k_means.compute_cluster_error(clusters[i], centers[i])))
            # print('\n')
            temp.append(compute_cluster_error(clusters[i], centers[i]))
        errors.append(np.mean(temp))

    k_values = [i for i in range(1, 16)]

    draw_clustering_error(k_values, errors)