from scipy.spatial.distance import cdist
from collections import Counter

def knn_predict(X_train, y_train, X_test, k):
    """
    steps are as follows:
    1. compute full distance matrix using cdist and then apply .tolist()
    2. for each test row's distance list:
        a. pair each distance with its train index via enumerate
        b. sort pairs by distance in ascending order, take first k
        c. pull the labels for those k indices from y_train
        d. counter them, sort items by (-count, label) - pick[0][0]
    3. append to results, return results
    """

    distance_matrix = cdist(X_test, X_train, metric='euclidean').tolist()

    predictions = []

    for row in distance_matrix:
        neighbors = sorted(enumerate(row), key=lambda pair: pair[1])[:k]
        neighbors_labels = [y_train[idx] for idx, _ in neighbors]

        counts = Counter(neighbors_labels)
        winner = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        predictions.append(winner)

    return predictions

X_train = [[1, 1], [2, 1], [5, 4], [6, 5], [1, 5]]
y_train = ['A', 'A', 'B', 'B', 'C']
X_test  = [[2, 2], [5, 5]]
k = 3

print(knn_predict(X_train, y_train, X_test, k))