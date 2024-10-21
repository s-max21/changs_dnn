from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Random Forest Regression
def train_rf(x, y, rf_param_grid, ps):
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(rf, rf_param_grid, cv=ps, scoring='neg_mean_squared_error')
    grid_search.fit(x, y)

    rf_model = grid_search.best_estimator_
    return rf_model

def generate_trees(n):
    return [100 + j*30 for j in range(n)]