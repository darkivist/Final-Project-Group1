import keras_tuner as kt
import torch
import torch.nn as nn


class MWPSolver(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
#insert lstm model here

#technique sourced from tensorflow documentation: https://www.tensorflow.org/tutorials/keras/keras_tuner
#create class to allow keras tuner to build the best model from our pre-defined options
def model_builder(hp):
    num_layers = hp.Int('num_layers', 1, 4)
    hidden_size = hp.Int('hidden_size', 32, 256, step=32)
    dropout = hp.Float('dropout', 0, 0.5, step=0.1)

    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    model = MWPSolver(input_size, hidden_size, output_size)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model

#invoke keras hyperband tuner to set the objective for our training, lowest loss, and the folder to store our tuning data.
#should we actually do this for val_accuracy, val_loss instead? "objective" tells the tuner what to optimize for.

tuner = kt.Tuner(
    oracle=kt.tuners.Hyperband(
        objective='val_loss',
        max_trials=10))

#add early stopping callback to abandon hyperparameter tuning if model doesn't improve and move to next test
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

for epoch in range(100):
    with tuner.scoped_session():
        hp = tuner.oracle.get_state()
        model = model_builder(hp)

        loss = train_model(model)

        tuner.oracle.update_trial(trial_id, {'loss': loss})

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

print(f"""
Best model: {best_model}  
Best hyperparameters: {best_hyperparameters}
""")