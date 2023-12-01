import tensorflow as tf
import json
from helper import read_from_file, decode_weights, ReadandProcessData, write_to_file, encode_weights
import numpy as np
from sklearn.metrics import accuracy_score
import sys
from tensorflow.keras.models import model_from_json

best_accuracy = sys.argv[1]
weights_arr = read_from_file("received_updated_weights.txt")
weights_arr = json.loads(weights_arr)

average_weights = {}
for i in range(len(weights_arr)):
    weights_arr[i] = json.loads(weights_arr[i])
    keys_arr = weights_arr[i].keys()
    for layer_name in keys_arr:
        if layer_name == "number_of_train":
            continue
        weights_arr[i][layer_name] = decode_weights(weights_arr[i][layer_name])

# load model weights
model_weights = decode_weights(read_from_file("model_weights.txt"))
# load model from json
model = model_from_json(read_from_file("model_config.json"))
# compile the model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# set weights to the model
model.set_weights(model_weights)

# Counting the number of data points
total_data_points = 0
for i in range(len(weights_arr)):
    total_data_points += weights_arr[i]["number_of_train"]

# updating weights in each layer
index = 0
for layer in model.layers:
    if layer.trainable_weights:
        # print( len(weights_arr[0][layer.name]))
        # weights_arr[0][layer.name] = np.array(weights_arr[0][layer.name])
        for j in range(0,len(weights_arr[0][layer.name])):
            weights_arr[0][layer.name][j] *= (weights_arr[0]["number_of_train"]/total_data_points)
        # temp_list = np.array(weights_arr[0][layer.name],dtype=float)
        # print(temp_list)
        # temp_list /= weights_arr[0]["number_of_train"]/total_data_points
        # weights_arr[0][layer.name] /= (weights_arr[0]["number_of_train"]/total_data_points)
        for i in range(1,len(weights_arr)):
            for j in range(0,len(weights_arr[0][layer.name])):
                weights_arr[0][layer.name][j] += ((weights_arr[i]["number_of_train"]/total_data_points) * weights_arr[i][layer.name][j])
        model.layers[index].set_weights(weights_arr[0][layer.name])
    index += 1

X_test,y_test = ReadandProcessData("test")
y_pred = model.predict(X_test,verbose=0)

# Test loss Calculation
y_true = []
for i in range(len(y_test)):
    newArr = [0,0,0,0,0,0,0,0,0,0,0,0]
    newArr[y_test[i]] = 1
    y_true.append(newArr)

cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(y_true, y_pred).numpy()

with open('val_loss.txt','a') as file:
    content_to_append = str(loss)+","
    file.write(content_to_append)

# Test prediction Accuracy 
y_pred = [np.argmax(ele) for ele in y_pred]
accuracyScore = accuracy_score(y_test,y_pred)*100
print(accuracy_score(y_test,y_pred)*100)
best_accuracy = (float)(best_accuracy)
if(accuracyScore > best_accuracy):
    model.save("best_model.h5")

# Global train loss Calculation
X_test,y_test = ReadandProcessData("global_train")
y_pred = model.predict(X_test,verbose=0)
y_true = []
for i in range(len(y_test)):
    newArr = [0,0,0,0,0,0,0,0,0,0,0,0]
    newArr[y_test[i]] = 1
    y_true.append(newArr)

cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(y_true, y_pred).numpy()

with open('global_loss.txt','a') as file:
    content_to_append = str(loss)+","
    file.write(content_to_append)

# Saving model weights
write_to_file('model_weights.txt',encode_weights(model.get_weights()))