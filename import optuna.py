import optuna

def objective(trial,estimator):
    window_size_in = 30
    window_size = 30
    if estimator== "CNN3":
        num_kernels1 = trial.suggest_int('num_kernels1', 32, 128)
        num_kernels2 = trial.suggest_int('num_kernels2', 32, 128)
        num_kernels3 = trial.suggest_int('num_kernels3', 32, 128)
        kernel_size  = trial.suggest_int('kernel_size', 3, 7)
    elif estimator== "CNN2_LSTM":
        num_kernels1 = trial.suggest_int('num_kernels1', 32, 128)
        num_kernels2 = trial.suggest_int('num_kernels2', 32, 128)
        kernel_size = trial.suggest_int('kernel_size', 3, 7)
        num_hiddens = trial.suggest_int('num_hiddens', 32, 128)
    else:
        num_hiddens1 = trial.suggest_int('num_hiddens1', 32, 128)
        num_hiddens2 = trial.suggest_int('num_hiddens2', 32, 128)
        num_hiddens3 = trial.suggest_int('num_hiddens3', 32, 128)
    

    alpha   = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
    l1_coef = trial.suggest_loguniform('l1_coef', 1e-5, 1e-2)

    # Create the model with the suggested hyperparameters
    if estimator == "CNN3":
        model = CNN3(num_kernels1=num_kernels1, num_kernels2=num_kernels2, num_kernels3=num_kernels3,
                        kernel_size=kernel_size, window_size_in=window_size_in, window_size=window_size,
                        alpha=alpha, l1_coef=l1_coef)
        history = model.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    elif estimator == "CNN2_LSTM":
        model = CNN2_LSTM(num_hiddens=num_hiddens, num_kernels1=num_kernels1, num_kernels2=num_kernels2,
                        kernel_size=kernel_size, window_size_in=window_size_in, window_size=window_size,
                        alpha=alpha, l1_coef=l1_coef)
        history = model.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    elif estimator == "LSTM3":
        model = LSTM3(num_hiddens1=num_hiddens1, num_hiddens2=num_hiddens2, num_hiddens3=num_hiddens3,
                        window_size_in=window_size_in, window_size=window_size,
                        alpha=alpha, l1_coef=l1_coef)
        history = model.fit(X_CNN, Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    else:
        # For NN3
        model = NN3(num_hiddens1=num_hiddens1, num_hiddens2=num_hiddens2, num_hiddens3=num_hiddens3,
                        window_size_in=window_size_in, window_size=window_size,
                        alpha=alpha, l1_coef=l1_coef)
        history = model.fit(X_CNN.reshape(17099,30*11), Y_CNN, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    val_loss = history.history['val_loss'][-1]
    # val_mae = history.history['val_mae'][-1]

    return val_loss


NNt = {"NN3":NN3, "LSTM3":LSTM3, "CNN3":CNN3, "CNN2_LSTM":CNN2_LSTM}

print(f"Network is {"CNN2_LSTM"}")
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, estimator="CNN2_LSTM"), n_trials=50)
print("Best hyperparameters: ", study.best_params)
print("Best trial: ", study.best_trial)
print("Best value: ", study.best_value)