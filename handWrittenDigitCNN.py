import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, input_shape=(28, 28, 1), strides=(2,2), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, strides=(2,2), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides=(2, 2), padding='same'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

result = model.evaluate(x_test, y_test)
print(f"accuracy : {result[1]*100:0.2f}%, loss : {result[0]*100:0.2f}%")

model.save("./checkpoints/handWrittenCNN.h5")