import tensorflow as tf
from layers import L1Dist
from tensorflow.keras.models import Model #one of the most important layers
"""Model(inputs=inputImage verificationImage, outputs = 0 or 1); allows you to define"""
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, BatchNormalization, Dropout #different deep learning layer types
def make_embedding():
    # Input layer with corrected name
    inp = Input(shape=(100,100,3), name='input_image')

    # First Block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    bn1 = BatchNormalization()(c1)
    # MaxPooling with corrected filter size (should be tuple not int)
    m1 = MaxPooling2D((2,2), padding='same')(bn1)

    # Second Block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    bn2 = BatchNormalization()(c2)
    # Corrected MaxPooling
    m2 = MaxPooling2D((2,2), padding='same')(bn2)

    # Third Block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    bn3 = BatchNormalization()(c3)
    # Corrected MaxPooling
    m3 = MaxPooling2D((2,2), padding='same')(bn3)

    # Final Embedding Block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    bn4 = BatchNormalization()(c4)
    f1 = Flatten()(bn4)
    # Add dropout to reduce overfitting
    dr1 = Dropout(0.3)(f1)
    d1 = Dense(4096, activation='sigmoid')(dr1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')
def make_siamese_model():

    # Anchor image input
    input_image = Input(name = 'input_image', shape =(100,100,3))

    # Validation Image in network
    validation_image = Input(name='validation_img',shape = (100,100,3))

    # combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'

    embedding = make_embedding()
    # pass input images into siamese distance layer - outputs 4096 length vector
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification Layer: combine final distances into finally fully connected layer with sigmoid activation, outputs a 1 or 0
    classifier = Dense(1, activation = 'sigmoid')(distances)

    return Model(inputs = [input_image,validation_image], outputs = classifier, name='SiameseNetwork')


