import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D,SeparableConv2D, Input, Conv2D, BatchNormalization,Add,GlobalAveragePooling2D,Lambda,Dense
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from helper import encode_weights,write_to_file
from keras.utils import plot_model

def residual_block(x, channels, stride=1, weight_decay=5e-4,max_pool=True):
    ki = initializers.he_normal(seed=9)
    kr = regularizers.l2(weight_decay)
    
    
    x1 = SeparableConv2D(channels, (3, 3), kernel_initializer=ki, strides=(stride, stride),
               use_bias=False, padding='same', kernel_regularizer=kr)(x)
    x1 = BatchNormalization()(x1)
    x1 = Lambda(lambda x: x * tf.math.tanh(tf.math.log(1 + tf.exp(x))))(x1)
    
    x2 = SeparableConv2D(channels, (5, 5), kernel_initializer=ki, strides=(stride, stride),
               use_bias=False, padding='same', kernel_regularizer=kr)(x)
    x2 = BatchNormalization()(x2)
    x2 = Lambda(lambda x: x * tf.math.tanh(tf.math.log(1 + tf.exp(x))))(x2)

    x = Add()([x, x1,x2])
    
    if max_pool:
        x = MaxPooling2D()(x)
    
    return x

def create_custom_model(inp, num_classes):
    x = inp
    
    x = Conv2D(128, (5, 5), strides=2, kernel_initializer=initializers.he_normal(),
               kernel_regularizer=regularizers.l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: x * tf.math.tanh(tf.math.log(1 + tf.exp(x))))(x)
    
    for i in range(0,4):
        if i == 3:
            x = residual_block(x, channels=128,max_pool = False)
        else :
            x = residual_block(x, channels=128)
    
    # x = Conv2D(256, (1, 1), strides=(1,1), kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(5e-4), use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Lambda(lambda x: x * tf.math.tanh(tf.math.log(1 + tf.exp(x))))(x)
    x = SeparableConv2D(num_classes, (1, 1), strides=1, kernel_initializer=initializers.he_normal(),kernel_regularizer=regularizers.l2(5e-4), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: x * tf.math.tanh(tf.math.log(1 + tf.exp(x))))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs=inp, outputs=x)
    
    return model

input_shape = Input((224, 224, 3))
num_classes = 12
model = create_custom_model(input_shape, num_classes)

print(model.summary())
# configuration of the model
model_config = model.to_json()
# Write the JSON string to a file
write_to_file('model_config.json', model_config)

# encoding weights of the model to base64 string and writing to a file
encoded_weight = encode_weights(model.get_weights())  
write_to_file('model_weights.txt',encoded_weight)