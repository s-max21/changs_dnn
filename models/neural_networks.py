import keras_tuner as kt
from tensorflow import keras

# Define the model-building function for Keras Tuner
def neural_net(shape, n_layers, min_units, max_units, step):
    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=shape))

        # Tune the number of units (neurons) for all layers
        n_units = hp.Int('units', min_value=min_units, max_value=max_units, step=step)

        for _ in range(n_layers):
            model.add(keras.layers.Dense(n_units, activation='relu'))

        # Output layer for regression
        model.add(keras.layers.Dense(1))

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse'])

        return model

    return build_model


def train_neural_net(model, x_train, y_train, x_test, y_test, project_name, directory):
        tuner = kt.GridSearch(
            model,
            objective='val_loss',  # Target metric
            max_trials=5,  # Number of hyperparameter sets to try
            executions_per_trial=1,  # Number of models to build for each trial
            directory=directory,
            project_name=project_name)
        
        # Perform the search
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        tuner.search(x_train, y_train,
                      epochs=1000,
                      validation_data=(x_test, y_test),
                      callbacks=[early_stopping])

        # Retrieve the best model and hyperparameters
        nn_model = tuner.get_best_models(num_models=1)[0]
        return nn_model