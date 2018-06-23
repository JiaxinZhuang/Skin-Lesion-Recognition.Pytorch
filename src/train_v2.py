from keras.preprocessing import image
from keras.applications import resnet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import SGD
from keras import losses
import numpy as np

batch_size = 32
num_classes = 7

base_model = resnet50.ResNet50

base_model = base_model(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                        optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0),
                        metrics=['categorical_accuracy', 'acc'])

    x_train = np.random.normal(loc=127, scale=127, size=(50, 224,224,3))
    y_train = np.array([0,1]*25)
    x_train = resnet50.preprocess_input(x_train)

    print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0))
        model.fit(x_train, y_train,
                          epochs=100,
                                    batch_size=batch_size,
                                              shuffle=False,
                                                        validation_data=(x_train, y_train))
