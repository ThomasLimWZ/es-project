import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import ParameterGrid

# Convolutional Neural Network
def cnn_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, model=None, epochs = 50,  optimizer = 'adam', batch_size = 32):
    # Get the number of unique classes
    num_classes = y_train.nunique()
    print('Number of classes: ', num_classes)
    # Determine the appropriate loss function and number of units in the last layer
    if num_classes == 2:
        loss = 'binary_crossentropy'
        output_units = 1
        last_activation = 'sigmoid'
    else:
        loss = 'sparse_categorical_crossentropy'
        output_units = num_classes
        last_activation = 'softmax'

    # If no model is provided, create a new one
    if model is None:
        model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
        ])
    
    # add last layer
    model.add(Dense(output_units, activation=last_activation))
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=2)
    valid_loss, valid_accuracy = model.evaluate(X_valid, y_valid, verbose=2)
    
    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {valid_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    # Make predictions on the test set
    y_pred_cnn = model.predict(X_test)
    
    # Convert the predictions to class labels
    if loss != 'binary_crossentropy':
        y_pred_cnn = np.argmax(y_pred_cnn, axis=1)
    else:
        y_pred_cnn = (y_pred_cnn > 0.5).astype(int)

    return model, y_pred_cnn, test_accuracy, history

# Multilayer Perceptron
def mlp_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, params = { 'hidden_layer_sizes': (100, 100, 100), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000, 'random_state': 42}):
    model = MLPClassifier(**params)
    
    # Fit the model
    model = model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_valid = model.predict(X_valid)
    
    # Accuracy
    model_accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print('Train Accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    print('Validation Accuracy: ', accuracy_score(y_valid, y_pred_valid), '\n')
    print(classification_report(y_test, y_pred))

    return model, y_pred, model_accuracy

# Support Vector Machine
def svm_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, params = {'C': 1.0, 'kernel': 'linear', 'random_state':42, 'probability': True, 'gamma': 'scale', 'decision_function_shape': 'ovr'}):
    # Create the SVM model
    model = SVC(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_valid = model.predict(X_valid)
    
    # Accuracy
    model_accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print('Train Accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    print('Validation Accuracy: ', accuracy_score(y_valid, y_pred_valid), '\n')
    
    print(classification_report(y_test, y_pred))

    return model, y_pred, model_accuracy

# Decision Tree
def dt_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, params = {'criterion': 'gini', 'max_depth': 20, 'random_state':42, 'splitter': 'best'}):
    # Create the Decision Tree model
    model = DecisionTreeClassifier(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_valid = model.predict(X_valid)
   
    # Accuracy
    model_accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print('Train Accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    print('Validation Accuracy: ', accuracy_score(y_valid, y_pred_valid), '\n')
    print(classification_report(y_test, y_pred))

    return model, y_pred, model_accuracy

# Random Forest
def rf_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, params = {'n_estimators':100, 'random_state':42}):
    # Create the Random Forest model
    model = RandomForestClassifier(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_valid = model.predict(X_valid)
    
    # Accuracy
    model_accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print('Train Accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    print('Validation Accuracy: ', accuracy_score(y_valid, y_pred_valid), '\n')
    print(classification_report(y_test, y_pred))

    return model, y_pred, model_accuracy

# LightGBM
def lgbm_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, params = {'n_estimators':1000, 'random_state':42, 'learning_rate': 0.1, 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1}):
    # Create the LightGBM model
    model = lgb.LGBMClassifier(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_valid = model.predict(X_valid)

    # Accuracy
    model_accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    print('Train Accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    print('Validation Accuracy: ', accuracy_score(y_valid, y_pred_valid), '\n')
    print(classification_report(y_test, y_pred))

    return model, y_pred, model_accuracy

# XGBoost
def xgboost_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, params = {'n_estimators':100, 'random_state':42}):
    # Create the XGBoost model
    model = XGBClassifier(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_valid = model.predict(X_valid)

    # Accuracy
    model_accuracy = accuracy_score(y_test, y_pred)

    # Classification report
    print('Train Accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    print('Validation Accuracy: ', accuracy_score(y_valid, y_pred_valid), '\n')
    print(classification_report(y_test, y_pred))

    return model, y_pred, model_accuracy

# CatBoost
def catboost_classifier(X_train, X_test, X_valid, y_train, y_test, y_valid, params = {'n_estimators':100, 'random_state':42}):
    # Create the CatBoost model
    model = CatBoostClassifier(**params)

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_valid = model.predict(X_valid)

    # Accuracy
    model_accuracy = accuracy_score(y_test, y_pred)

    # Classification report
    print('Train Accuracy: ', accuracy_score(y_train, model.predict(X_train)))
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    print('Validation Accuracy: ', accuracy_score(y_valid, y_pred_valid), '\n')
    print(classification_report(y_test, y_pred))

    return model, y_pred, model_accuracy

# Hyperparameter Tuning
def hyperparameter_tuning(classifier, param, X_train, y_train, X_test, y_test, verbose = 1):
    model_constructors = {
        'mlp': MLPClassifier,
        'svm': SVC,
        'dt': DecisionTreeClassifier,
        'rf': RandomForestClassifier,
        'lgbm': lgb.LGBMClassifier,
        'xgb': XGBClassifier,
        'cat': CatBoostClassifier
    }

    if classifier not in model_constructors:
        raise Exception('Incorrent classifier. Please make sure it is within the options')
    
    best_model = None
    best_params = None
    best_y_pred = None
    best_acc = 0

    # Create all possible combinations of hyperparameters
    param_combinations = ParameterGrid(param)

    # Iterate over all combinations of hyperparameters
    for params in param_combinations:
        # Create a new classifier with the current set of hyperparameters
        clf = model_constructors[classifier]()
        clf.set_params(**params)

        # Fit the classifier
        clf.fit(X_train, y_train)
        
        # Evaluate the classifier
        pred_test = clf.predict(X_test)
        acc = accuracy_score(y_test, pred_test)

        # Print the hyperparameters and testing accuracy if verbose more than 0
        if verbose > 0:
            print("Hyperparameters:", params)
            print(f"Testing Accuracy: {acc}\n")
        
        # Update the best hyperparameters and model as well as the best accuracy
        if acc > best_acc:
            best_model = clf
            best_params = params
            best_acc = acc
            best_y_pred = pred_test

    # Print the best hyperparameters and testing accuracy
    if verbose > 0:
        print("Best Hyperparameters:", best_params)
        print("Best Testing Accuracy:", best_acc)

    return best_model, best_params, best_y_pred, best_acc

def cnn_classifier_hp(X_train, X_test, X_valid, y_train, y_test, y_valid, epochs = 50, dataset_num = 1, batch_size = 32):
    # To build the new model with the best hyperparameters
    if dataset_num == 1:
        directory = 'dir'
        project_name = 'cnn_hyperparameter_tuning'
    else:
        directory = f'dir{dataset_num}'
        project_name = f'cnn_hyperparameter_tuning{dataset_num}'
    
    # Get the number of unique classes
    num_classes = y_train.nunique()
    input_shape = (X_train.shape[1], 1)

    # Initialize the tuner
    tuner = kt.Hyperband(
        lambda hp: build_cnn_model(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_epochs=25,
        factor=3,
        directory=directory,
        project_name=project_name
    )

    # Perform hyperparameter tuning
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the best hyperparameters
    model = build_cnn_model(best_hps, input_shape, num_classes)

    # Train the best model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=2)
    valid_loss, valid_accuracy = model.evaluate(X_valid, y_valid, verbose=2)

    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {valid_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    # Make predictions on the test set
    y_pred_cnn = model.predict(X_test)

    if num_classes > 2:
        y_pred_cnn = np.argmax(y_pred_cnn, axis=1)
    else:
        y_pred_cnn = (y_pred_cnn > 0.5).astype(int)
        
    return model, best_hps, y_pred_cnn, test_accuracy, history

# CNN model for hyperparameter tuning
def build_cnn_model(hp, input_shape, num_classes):
    """
    Build a Convolutional Neural Network (CNN) model for hyperparameter tuning.

    Parameters:
    - hp (keras_tuner.HyperParameters): Hyperparameters object for tuning.
    - input_shape (tuple): Shape of the input data (excluding batch size).
    - num_classes (int): Number of classes in the classification task.

    Hyperparameters:
    - l2_value (float): L2 regularization strength.
    - filters_1 (int): Number of filters in the first convolutional layer.
    - kernel_size_1 (int): Size of the kernel in the first convolutional layer.
    - filters_2 (int): Number of filters in the second convolutional layer.
    - kernel_size_2 (int): Size of the kernel in the second convolutional layer.
    - dense_units (int): Number of units in the dense layer.
    - learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
    - model (keras.models.Sequential): Compiled CNN model.
    """ 
    
    # Define l2 regularization
    l2_value = hp.Float('l2_value', min_value=1e-5, max_value=1e-2, sampling='LOG')
    
    # Determine the appropriate loss function and number of units in the last layer
    if num_classes == 2:
        loss = 'binary_crossentropy'
        output_units = 1
        last_activation = 'sigmoid'
    else:
        loss = 'sparse_categorical_crossentropy'
        output_units = num_classes
        last_activation = 'softmax'
        
    # print(loss)
    # print(num_classes)
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(
            filters=hp.Int('filters_1', min_value=32, max_value=512, step=32),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
            activation='relu',
            padding='same',
            kernel_regularizer=l2(l2_value)
        ),
        Flatten(),
        Dense(
            units=hp.Int('dense_units_1', min_value=32, max_value=128, step=32),
            activation= hp.Choice('activation_1', ['relu', 'tanh', 'sigmoid']),
            kernel_regularizer=l2(l2_value)
        ),       
        Dense(
            units=hp.Int('dense_units_2', min_value=32, max_value=128, step=32),
            activation= hp.Choice('activation_2', ['relu', 'tanh', 'sigmoid']),
            kernel_regularizer=l2(l2_value) 
        ),
        Dense(output_units, activation = last_activation)
    ])

    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss=loss,
        metrics=['accuracy']
    )
    return model