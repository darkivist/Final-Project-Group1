import keras_tuner as kt
from tensorflow import keras
import torch
import torch.nn as nn


class MWPSolver(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
#insert lstm model here or fold into model_builder class instead?

#technique sourced from tensorflow documentation: https://www.tensorflow.org/tutorials/keras/keras_tuner
#create class to allow keras tuner to build the best model from our pre-defined options
#should we expand the parameters we want to tune, or scale back?
def model_builder(hp):
    num_layers = hp.Int('num_layers', 1, 4)
    hidden_size = hp.Int('hidden_size', 32, 256, step=32)
    dropout = hp.Float('dropout', 0, 0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model = MWPSolver(input_size, hidden_size, output_size) #or build model in this function instead?
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model

#invoke keras hyperband tuner to set the objective for our training, lowest loss, and the directory to store our tuning data.
#should we actually do this for val_accuracy, val_loss instead? "objective" tells the tuner what to optimize for.
#or another metric?

tuner = kt.Tuner(
    oracle=kt.tuners.Hyperband(
        objective='val_loss',
        max_trials=10,
        factor=3,#higher is faster
        directory='lstm_hyperparameter_tuning',
        project_name='final_project'))

#add early stopping callback to abandon hyperparameter tuning if model doesn't improve and move to next test
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#perform hyperparameter search
tuner.search(x=X_train, y=y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[stop_early])

#save best model and parameters
best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

print(f"""
The hyperparameter search is complete.
Best LSTM Model: {best_model} 
Best Hyperparameters:
- Learning Rate: {best_hyperparameters.get('learning_rate')}
- Num Layers: {best_hyperparameters.get('num_layers')}  
- Dropout: {best_hyperparameters.get('dropout')}
- Hidden Size: {best_hyperparameters.get('hidden_size')}
""")

#now train for best epoch
model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(x=X_train, y=y_train, epochs=10, validation_data=(X_val, y_val))

val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#retrain with best hyperparameters and epoch
hypermodel = tuner.hypermodel.build(best_hyperparameters)
history = hypermodel.fit(x=X_train, y=y_train, epochs=best_epoch, validation_data=(X_val, y_val))

#hypermodel ends up being our model to run on the word problems data