import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Input, Dense, Concatenate
from keras.initializers import RandomUniform, Zeros
from keras.regularizers import L2
import numpy as np
import concurrent.futures



class TruncateLayer(Layer):
    """
    A Keras layer that truncates every input to a specified range.

    Parameters
    ----------
    beta : float
        The value used to determine the truncation range.
        The inputs will be clipped to the range [-beta, beta].

    Methods
    -------
    call(inputs)
        Applies truncation to the inputs based on the `beta` value.

    """

    def __init__(self, beta):
        super().__init__()
        self.beta = tf.constant(beta, dtype=tf.float32)

    def call(self, inputs):
        return tf.clip_by_value(
            inputs, clip_value_min=-self.beta, clip_value_max=self.beta
        )



def create_network(n=100, d=1, n_units=10, n_layers=10, network_id=0):
    """
    Creates a neural network with n_layers hidden layers.

    Parameters
    ----------
    n: int
        number of training samples
    d: int
        dimension of input variables
    n_units: int, optional
        number of neurons in each hidden layer
    n_layers: int, optional
        number of hidden layers

    Returns
    -------
    model: keras.models.Sequential
        Sequential model containing a truncation layer as last layer

    """

    # Define submodel
    model = keras.models.Sequential(name=f"subnetwork_{network_id}")

    # Create input layer
    model.add(
        Dense(
            units=n_units,
            activation="sigmoid",
            kernel_initializer=RandomUniform(minval=-n**((1/d+1)/2), maxval=n**((1/d+1)/2)),
            bias_initializer=RandomUniform(minval=-n**((1/d+1)/2), maxval=n**((1/d+1)/2)),
            name=f"input_layer_{network_id}"
        )
    )

    # Create n_layers-1 hidden layers
    for i in range(n_layers - 1):
        model.add(
            Dense(
                units=n_units,
                activation="sigmoid",
                kernel_initializer=RandomUniform(minval=-20*d*tf.math.log(n)**2, maxval=20*d*tf.math.log(n)**2),
                bias_initializer=RandomUniform(minval=-20*d*tf.math.log(n)**2, maxval=20*d*tf.math.log(n)**2),
                name=f"hidden_layer_{network_id}_{i}"
            )
        )

    # Create output layer
    model.add(
        Dense(
            units=1,
            activation="sigmoid",
            kernel_initializer=RandomUniform(minval=-20*d*tf.math.log(n)**2, maxval=20*d*tf.math.log(n)**2),
            bias_initializer=RandomUniform(minval=-20*d*tf.math.log(n)**2, maxval=20*d*tf.math.log(n)**2),
            name=f"output_layer_{network_id}"
        )
    )

    return model


def create_dnn(
    train_shape,
    n_networks=100,
    n_layers=10,
    n_units=10,
    c=0.0001
):
    """
    Creates a model with n_networks subnetworks, each consisting of n_layers hidden layers.
    The output is the output of the subnetworks combined in a last dense layer.

    Parameters
    ----------
    train_shape: tuple
        Shape of the training data.
    n_networks: int, optional
        Number of subnetworks to train.
    n_layers: int, optional
        Number of hidden layers in each subnetwork.
    n_units: int, optional
        Number of neurons in each hidden layer.
    c: float, optional
        Regularization parameter.

    Returns
    -------
    model: keras.models.Model
        The created model with the specified structure.
    """

    # Define input shape based on dimension of input variable
    n, d = train_shape
    input_shape = (d,)
    n, d = float(n), float(d)

    # Create a list containing n_networks DNNs with n_layers hidden layers
    def build_network_parallel():
        # Create a list containing n_networks DNNs with n_layers hidden layers in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(create_network, n, d, n_units, n_layers, network_id) for network_id in range(n_networks)]
            sub_networks = [future.result() for future in concurrent.futures.as_completed(futures)]
        return sub_networks
    
    sub_networks = build_network_parallel()

    # Create the output layer
    output_layer = Dense(
        units=1,
        kernel_regularizer=L2(c),
        use_bias=False,
        kernel_initializer=Zeros(),
        name = "output_layer"
    )

    # Define the structure of the combined model
    inputs = Input(shape=input_shape, name="input_layer")
    kn_outputs = [sub_net(inputs) for sub_net in sub_networks]
    concatenated_outputs = Concatenate()(kn_outputs)
    outputs = output_layer(concatenated_outputs)

    # Create the model
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name = "dnn"
    )

    return model


def indicator_function(x, a):
    boolean = (x >= -a) & (x <= a)
    return np.all(boolean, axis=1).astype(int)


def custom_loss_function(a, x_train):
    def loss(y_true, y_pred):
        loss_value = tf.square(y_true - y_pred)
        loss = tf.reduce_mean(loss_value * indicator_function(x_train, a))
        return loss
    return loss


def dataset(X, y, batch_size=32, shuffle=False):
    """
    Converts features and labels into a tf.data.Dataset, 
    with optional shuffling and batching.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle only if it's the training data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    # Batch and prefetch the dataset for performance optimization
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch


# Keras Tuner HyperModel Definition
class MyHyperModel(HyperModel):
    def __init__(self, x_train, a):
        self.x_train = x_train
        self.n = x_train.shape[0]
        self.a = a

    def build(self, hp):
        # Hyperparameter-Bereich definieren
        n_networks = hp.Int('n_networks', min_value=self.n, max_value=2*self.n, step=self.n//2)
        n_units = hp.Int('n_units', min_value=5, max_value=20, step=5)
        n_layers = hp.Int('n_layers', min_value=5, max_value=20, step=5)
        learning_rate = hp.Float("learning_rate", min_value=1e-6, max_value=1e-4, sampling="log")

        # Modell erstellen
        model = create_dnn(train_shape=self.x_train.shape,
                            n_networks=n_networks,
                              n_units=n_units,
                                n_layers=n_layers)

        # Optimizer und Loss Function
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                       loss=custom_loss_function(self.a, self.x_train),
                         metrics=['mse'])

        return model
    

def train_dnn(x_train, y_train, x_test, y_test, project_name, directory, a=1):
    # Random Search Tuner
    hypermodel = MyHyperModel(x_train=x_train, a=a)

    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=5,  # Anzahl der Modelle, die ausprobiert werden
        executions_per_trial=1,  # Wie oft jedes Modell trainiert wird
        directory=directory,
        project_name=project_name)
    
    train_data = dataset(x_train, y_train)
    test_data = dataset(x_test, y_test)

    # Tuning starten
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    tuner.search(train_data,
                  epochs=1000,
                  validation_data=test_data,
                  callbacks=[early_stopping])

    # Bestes Modell abrufen
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model