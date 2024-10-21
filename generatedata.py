import sys
import os

# Add the 'src' directory to the Python path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath('generate_data.ipynb')))
src_path = os.path.join(project_dir, 'src')
sys.path.append(src_path)

# Now you can import the data module
from data import get_data, preprocess
import numpy as np
from sklearn.metrics import mean_squared_error

def regression_function(x):
    return np.exp(np.linalg.norm(x, ord=2, axis=1))

# Generate trsin, test and validation data
def generate_data(n_samples, sigma):
    # Generate the data
    mse = []
    for i in range(50):
        X, y = get_data(regression_function,
                        n_features=7,
                        n_samples=n_samples,
                        sigma=sigma,
                        omega=1.64,
                        seed=i)
        
        X_train, y_train, X_test, y_test = preprocess(X,y)
    
        X_val, y_val = get_data(regression_function,
                                n_features=7,
                                n_samples=10**5,
                                sigma=sigma,
                                omega=0.,
                                seed=i+50)
        
        y_avg = np.mean(y)
        error = mean_squared_error(regression_function(X), y_avg*np.ones(y.shape))
        mse.append(error)

        np.savez(os.path.join(project_dir, f'data/data_sigma{sigma}_samples{n_samples}_{i}.npz'),
                 X_train=X_train,
                 y_train=y_train,
                 X_test=X_test,
                 y_test=y_test,
                 X_val=X_val,
                 y_val=y_val)
        
    error_scaler = np.mean(mse)
    np.savez(os.path.join(project_dir, f'data/error_scaler_sigma{sigma}_samples{n_samples}.npz'),
             error_scaler=error_scaler)
    
generate_data(100, 0.05)
generate_data(100, 0.2)
generate_data(200, 0.05)
generate_data(200, 0.2)