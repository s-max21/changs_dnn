from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, PredefinedSplit
import numpy as np

# Function to calculate IQR
def calculate_iqr(data):

    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25


def iqr_median(my_func, n_features=1, n_samples=10**5, n_repetitions=100):
    """Function to calculate the IQR for a given function"""

    iqr_values = []
    # Generate random samples of x and calculate IQR for each set
    for _ in range(n_repetitions):
        x_samples = np.random.rand(n_samples, n_features)
        my_func_values = my_func(x_samples)
        iqr_values.append(calculate_iqr(my_func_values))

    return np.median(iqr_values)


# Generate random data samples
def get_data(my_func, n_features=1, n_samples=10**3, sigma=0.05, omega=None, seed=42):
    """Function to generate random data samples"""

    rng = np.random.default_rng(seed)

    if omega is None:
        omega = iqr_median(my_func, n_features=n_features)

    x = rng.random(size=(n_samples, n_features))
    y = my_func(x) + sigma * omega * rng.standard_normal(size=n_samples)

    return (x, y)


# Preprocess data by splitting into training and testing sets and scaling
def preprocess(x, y, seed=42):
    """Function to preprocess data for training"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    #scaler = StandardScaler()
    #scaler.fit(x_train)

    #x_train = scaler.transform(x_train)
    #x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def get_predefined_split(X_train, X_test):
    split_index = np.concatenate((np.full(len(X_train), -1),
                                  np.full(len(X_test), 0)))
    ps = PredefinedSplit(test_fold=split_index)
    return ps


def load_data(data_file=None):
    data = np.load(data_file)

    # Unpack the data
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    X_val = data['X_val']
    y_val = data['y_val']

    # Get predefined split
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    return X, y, X_train, y_train, X_test, y_test, X_val, y_val
