from sklearn.metrics import mean_squared_error

def evaluate(model, x_val, y_val):
    y_pred = model.predict(x_val)
    mse = mean_squared_error(y_pred, y_val)
    return mse
