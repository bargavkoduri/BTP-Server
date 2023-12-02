from tensorflow.keras.models import load_model
from helper import encode_weights, write_to_file

model = load_model("best_model.h5")

# configuration of the model
model_config = model.to_json()
# Write the JSON string to a file
write_to_file('model_config.json', model_config)

encoded_weight = encode_weights(model.get_weights())  # encoding weights of the model to base64 string and writing to a file
write_to_file('model_weights.txt',encoded_weight)