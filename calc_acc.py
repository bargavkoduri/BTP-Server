from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import model_from_json
from helper import read_from_file,decode_weights,ReadandProcessData

model = model_from_json(read_from_file("model_config.json"))
model_weights = decode_weights(read_from_file("model_weights.txt"))
model.set_weights(model_weights)

X_test,y_test = ReadandProcessData("test")
y_pred = model.predict(X_test,verbose=0)
y_pred = [np.argmax(ele) for ele in y_pred]
print(accuracy_score(y_test,y_pred)*100)