import joblib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network

# Train
def train(X_train, Y_train):
    
    # Create a model
    model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', solver='adam', 
                                                 alpha=0.0001, batch_size=1000, learning_rate='constant', learning_rate_init=0.001, power_t=0.9, 
                                                 max_iter=1000, shuffle=True, random_state=None, tol=0.00001, verbose=False, warm_start=False, momentum=0.9, 
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=20)

    # Train the model on the whole data set
    model.fit(X_train, Y_train)

    # Save the model (Make sure that the folder exists)
    joblib.dump(model, 'models\\mlp_classifier-opt.jbl')

    # Evaluate on training data
    predictions = model.predict(X_train)
    accuracy = sklearn.metrics.accuracy_score(Y_train, predictions)
    accuracy_val = 'Accuracy train: {0:.2f} %'.format(accuracy * 100.0)
    return accuracy_val

# Evaluate
def evaluate(X_test, Y_test):
    # Evaluate on test data

    model = joblib.load('models\\mlp_classifier-opt.jbl')
    predictions = model.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
    accuracy_val = 'Accuracy evaluate: {0:.2f} %'.format(accuracy * 100.0)
    return accuracy_val

def predict(X_test, input_predict):
    model = joblib.load('models\\mlp_classifier-opt.jbl')
    
    # Predict the Labels using the reloaded Model  
    predictions = model.predict(X_test)
    predict_val = 'Hasil prediksi data: '+ str(predictions[int(input_predict)])
    return predict_val