from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def train_knn(x, y, knn_param_grid, ps):
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(knn, knn_param_grid, cv=ps, scoring='neg_mean_squared_error')
    grid_search.fit(x, y)

    knn_model = grid_search.best_estimator_
    return knn_model

def generate_neighbors(n):
    u = [1, 2, 3]
    v = list(range(4, n + 1, 4))
    return u + v