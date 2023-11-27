import optuna
from train import train_model
from config import  get_config

def objective(trial):
    config = get_config()
    # Update the config with hyperparameters sampled by Optuna
    #config['batch_size'] = trial.suggest_int('batch_size', 4, 16)
    config['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    #config['d_model'] = trial.suggest_int('d_model', 256, 1024)
    config['num_epochs'] = trial.suggest_int('num_epochs', 0, 2)

    # Calling model
    validation_loss = train_model(config)
    print(validation_loss)

    return validation_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)