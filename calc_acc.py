from sklearn.metrics import accuracy_score
import numpy as np
from helper import ReadandProcessData
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")
X_test,y_test = ReadandProcessData("test")
y_pred = model.predict(X_test,verbose=0)
y_pred = [np.argmax(ele) for ele in y_pred]
print(accuracy_score(y_test,y_pred)*100)