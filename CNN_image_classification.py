# import tensorflow as tf
# from matplotlib import pyplot as plt
# import numpy as np
# import os
# import cv2
# import imghdr
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
#
# data_dir = 'Data'
# image_exts = ['jpeg', 'jpg', 'bmp', 'png']
#
# list_ = os.listdir(os.path.join(data_dir, 'happy'))
#
# print('Program is Running...')
# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         try:
#             img = cv2.imread(image_path)
#             tip = imghdr.what(image_path)
#             if tip not in image_exts:
#                 print(f"Image not in ext list {image_path}")
#                 os.remove(image_path)
#
#         except Exception as e:
#             print(f"Issue with image {image_path}")
#
# data = tf.keras.utils.image_dataset_from_directory('data')
# data_iterator = data.as_numpy_iterator()
# batch = data_iterator.next()
#
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])

# ===============================
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# (training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()
# training_images, testing_images = training_images/255, testing_images/255
#
# class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# for i in range(16):
#     plt.subplots()
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
#
# plt.show()

# training_images = training_images[:200]
# training_labels = training_labels[:200]
# testing_images = testing_images[:100]
# testing_labels = testing_labels[:100]

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(training_images, training_labels, epochs=20, validation_data=(testing_images, testing_labels))
#
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss : {loss} or Accuracy : {accuracy}")
#
# model.save('image_classifier.model')

# ------------
# model = tf.keras.models.load_model('image_classifier.model')
# img = cv.imread('dog.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
# plt.imshow(img, cmap=plt.cm.binary)
# prediction = model.predict(np.array([img])/255)
# index = np.argmax(prediction)
#
# print(f"Prediction is : {class_names[index]}")
# plt.show()
