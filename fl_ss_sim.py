from typing import List
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_array_ops import scatter_nd_non_aliasing_add_eager_fallback
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics.pairwise as smp
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, confusion_matrix
from fl_ss_data_processing import *
from csv import writer
import matplotlib.pyplot as plt
#from triangle_sector_similarity import Cosine_Similarity,Euclidean_Distance,TS_SS,Pairwise_TS_SS
import math
import torch
import csv
from itertools import zip_longest
import config
from keras.optimizers import gradient_descent_v2
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
#from tensorflow.python.ops.numpy_ops import np_config
import poison_config




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_sim_model(timesteps,n_features):
    # loading the saved model
    #loaded_model = tf.keras.models.load_model('./POISON_Persistent_Model/persistent_model_tf')
    #then call fit
    
    
    sgd = gradient_descent_v2.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    batch_size = poison_config.POISON_BATCH_SIZE
    model = Sequential()
    #model.add(LSTM(20, return_sequences=False, activation='tanh',input_shape=(timesteps, n_features)))

    model.add(Bidirectional(LSTM(100, return_sequences=False, activation='tanh'),input_shape=(timesteps, n_features)))
    model.add(Dense(50, activation='relu')) 
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    
    ####model.add(Bidirectional(LSTM(8, return_sequences=False, activation='tanh'),input_shape=(timesteps, n_features)))
    #model.add(Bidirectional(LSTM(8,batch_input_shape=(batch_size,timesteps, n_features))))
    ####model.add(Dense(8, activation='relu'))
    
    #model.add(Dense(2, activation='softmax'))
    #model.add(Dropout(.25))
    #model.add(LSTM(16))
    #model.add(Dense(units=1, activation='linear'))
    #model.add(Dropout(.25))
    #model.add(Dense(1, activation='softmax'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    #model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy(),'accuracy'])
    
    ### used to sgd
    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\model_summary.txt','a') as f:
    #        f.write(str(model.summary()))
    #        f.close()
    #print(model.summary())
    

    #return loaded_model
    return model

def model_sim_training(model,x_train,y_train,x_test,y_test,epochs):
    callbacks = EarlyStopping(monitor='binary_accuracy', mode='max', verbose=0, patience=1000,
                              restore_best_weights=True)
    checkpoint_filepath = './poison_epoch_models/POISON/best_model.h5'
    mc = ModelCheckpoint(filepath=checkpoint_filepath, monitor='binary_accuracy', mode='max', verbose=2, save_best_only=True)
    batch_size = 1
    X_train = x_train.copy()
    Y_train = y_train.copy()
    accuracy_callback = AccuracyCallback((X_train, Y_train))


    class_weights = {0:1.5,1:1.}

    #use verbose = 1 or 2 to see epoch progress pbar... each step is examples / batch
    train_history = model.fit(x_train,
                                y_train,
                                epochs=epochs,
                                validation_split=.2,
                                class_weight=class_weights,
                                shuffle=False,
                                #validation_data=(x_test, (y_test, x_test)),
                                #validation_data=(x_test, y_test),
                                batch_size=batch_size,
                                verbose=2,
                                callbacks=[callbacks,mc, accuracy_callback]
                                )
    #model.reset_states()
    print("\n\nBest Training Poisoning Accuracy:\n{}".format(max(train_history.history['binary_accuracy'])))
    with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_POISON_model_'+ config.LOG_NAME,'a') as f:
        f.write("\n\nBest Training Poisoning Accuracy:\n{}".format(max(train_history.history['binary_accuracy'])))
    f.close()
    model = load_model(checkpoint_filepath)
    #model.reset_states()

    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    ################ poison config was 1 - for 1 step prediction, i think i want the teimestps
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    print(yhat)
    model.reset_states()
    print("model.reset forcast")
    return yhat[0,0]

def model_sim_evaluate(path, attack, defense, log_name,model,x_train,y_train,x_test,y_test,epochs, num_sybils):
    '''
    test_labels = np.copy(y_test).astype("int32")
    for i in range(len(x_train)):
        X, y = x_train[i], y_train[i]
        yhat = forecast_lstm(model, 1, X)
        # forecast test dataset
        predictions = list()
        for i in range(len(x_test)):
            # predict
            X, y = x_test[i, 0:-1], y_test[i, -1]
            yhat = forecast_lstm(model, 1, X)
            # invert scaling
            #yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            #yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
        # report performance
        testAcc = accuracy_score(test_labels, predictions)
        print("here")
        print('%d) Test Accuracy: %.3f' % (i+1, testAcc))
    '''
    
    #scores = model.evalutate(x_train, y_train, config.POISON_BATCH_SIZE, verbose=0)
    #model.reset_states()
    #print("Model Accuracy: %.2f%%" %(scores[1]*100))
    train_pred = (model.predict(x_train, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,verbose=0) > .5).astype("int32") 
    #model.reset_states()
    train_labels = np.copy(y_train).astype("int32")
    test_pred = (model.predict(x_test) > .7).astype("int32") 
    #model.reset_states()
    test_labels = np.copy(y_test).astype("int32")
    print("predicted value:\n{}".format(test_pred))
    print("label value:\n{}".format(test_labels))
    trainAcc = accuracy_score(train_labels, train_pred)
    testAcc = accuracy_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred, zero_division=0)
    precision = precision_score(test_labels, test_pred)
    classes_report = classification_report(test_labels, test_pred)
    matrix = confusion_matrix(test_labels, test_pred, labels=[1,0])


    list_data = [epochs, testAcc,f1, precision]
    with open(config.PATH + config.ATTACK +'_'+ str(config.NUM_SYBILS) +'_sybil_'+ config.DEFENSE +'_poison_model_results.csv' ,'a',newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list_data)
        f_object.close()
    with open(path + attack +'_'+ str(num_sybils) +'_sybil_'+ defense +'_poison_model_'+ log_name,'a') as f:
            f.write('\n#####################         POISON         ###############################################\n')
            f.write('\n############################################################################################\n')
            f.write('\ncomm_round: {} | global_test_acc: {:.3%} | global_f1: {} | global_precision: {}\n'.format(epochs, testAcc, f1, precision))
            f.write(str(classes_report))
            f.write("\nAccuracy per class:\n{}\n{}\n".format(matrix,matrix.diagonal()/matrix.sum(axis=1)))
    f.close()
    print('\n#####################         POISON         ###############################################\n')
    print('\n############################################################################################\n')
    print('\ncomm_round: {} |global_train_acc: {:.3%}|| global_test_acc: {:.3%} | global_f1: {} | global_test_precision: {}'.format(epochs, trainAcc, testAcc, f1, precision))
    print(classes_report)
    print("\nAccuracy per class:\n{}\n{}\n".format(matrix,(matrix.diagonal()/matrix.sum(axis=1))))
   
classes = ['1.0','0.0']
class AccuracyCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_data):
        self.test_data = test_data
        self.class_history = ['1.0', '0.0']

    def on_epoch_end(self, epoch, logs=None):
        x_data, y_data = self.test_data

        correct = 0
        incorrect = 0

        x_result = self.model.predict(x_data, verbose=0)

        x_numpy = []

        for i in classes:
            self.class_history.append([])

        class_correct = [0] * len(classes)
        class_incorrect = [0] * len(classes)

        for i in range(len(x_data)):
            x = x_data[i]
            y = y_data[i]

            res = x_result[i]

            actual_label = np.argmax(y)
            pred_label = np.argmax(res)

            if(pred_label == actual_label):
                x_numpy.append(["cor:", str(y), str(res), str(pred_label)])     
                class_correct[actual_label] += 1   
                correct += 1
            else:
                x_numpy.append(["inc:", str(y), str(res), str(pred_label)])
                class_incorrect[actual_label] += 1
                incorrect += 1
        with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_poison_log.txt','a') as f:
                    f.write("\n\tCorrect: %d" %(correct))
                    f.write("\tIncorrect: %d" %(incorrect))
                    f.close()
        #print("\n\tCorrect: %d" %(correct))
        #print("\tIncorrect: %d" %(incorrect))

        for i in range(len(classes)):
            tot = float(class_correct[i] + class_incorrect[i])
            class_acc = -1
            if (tot > 0):
                class_acc = float(class_correct[i]) / tot
            with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_poison_log.txt','a') as f:
                        f.write("\t%s: %.3f" %(classes[i],class_acc))
                        f.close()
            #print("\t%s: %.3f" %(classes[i],class_acc)) 

        acc = float(correct) / float(correct + incorrect)  
        with open('C:\\Users\\ChristianDunham\\source\\repos\\Intrusion_Detection\\data\\output\\training_poison_log.txt','a') as f:
                    f.write("\tCurrent Network Accuracy: %.3f \n" %(acc))
                    f.close()
        #print("\tCurrent Network Accuracy: %.3f" %(acc))