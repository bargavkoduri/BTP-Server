from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras import regularizers
from tensorflow.keras.models import Model
from helper import encode_weights, write_to_file

# Load the EfficientNet B0 model with pre-trained weights
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers[:-5]:
    layer.trainable = False
    if hasattr(layer,'kernel'):
        layer.kernel_regularizer = regularizers.l2(0.01)
    if hasattr(layer,'bias'):
        layer.bias_regularizer = regularizers.l2(0.01)

model = base_model.output
model = Dropout(0.5)(model)
model = GlobalAveragePooling2D()(model)
model = Dense(12,activation='softmax')(model)

model = Model(base_model.input,model)

print(model.summary())

# configuration of the model
model_config = model.to_json()
# Write the JSON string to a file
write_to_file('model_config.json', model_config)

#  encoding weights of the model to base64 string and writing to a file
encoded_weight = encode_weights(model.get_weights())
write_to_file('model_weights.txt', encoded_weight)