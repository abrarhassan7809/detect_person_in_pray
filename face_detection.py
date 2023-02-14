import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import uuid

# =======dependencies=========
Model = tf.keras.models.Model
Layer = tf.keras.layers.Layer
Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
MaxPooling2D = tf.keras.layers.MaxPooling2D
Input = tf.keras.layers.Input
Flatten = tf.keras.layers.Flatten

# ========set gpu memory consumption growth===========
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_memory_growth(gpu, True)
    # tf.config.set_per_process_memory_growth(gpu, True)

# ==========create a folder path=====step-1=====
POS_PATH = os.path.join('data2', 'positive')
NEG_PATH = os.path.join('data2', 'negative')
ANC_PATH = os.path.join('data2', 'anchor')

if not os.path.isdir('data2'):
    os.makedirs('data2')
if not os.path.isdir(POS_PATH):
    os.makedirs(POS_PATH)
if not os.path.isdir(NEG_PATH):
    os.makedirs(NEG_PATH)
if not os.path.isdir(ANC_PATH):
    os.makedirs(ANC_PATH)
print('Folder created')
# # ==========move lfw images to negative folder=====step-2=======
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)):
#         EX_PATH = os.path.join('lfw', directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)
#
# print('All Down')
#
# # =========collect positive and anchor classes======step-3=======
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     # -----frame size 250x250-----
#     frame = frame[120:120+250, 200:200+250, :]
#
#     # -----collect anchor images------
#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         imgname = os.path.join(ANC_PATH, f'{uuid.uuid1()}.jpg')
#         cv2.imwrite(imgname, frame)
#
#     # -----collect positive images------
#     if cv2.waitKey(1) & 0xFF == ord('p'):
#         imgname = os.path.join(POS_PATH, f'{uuid.uuid1()}.jpg')
#         cv2.imwrite(imgname, frame)
#
#     cv2.imshow("Image Collection", frame)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.distroyAllWindows()


# ======================================
os.path.join(ANC_PATH, f"{uuid.uuid1()}.jpg")

# def data_aug(img):
#     data = []
#     for i in range(9):
#         img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
#         img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 3))
#         img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
#         img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100,
#                                                      seed=(np.random.randint(100), np.random.randint(100)))
#         img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1,
#                                                    seed=(np.random.randint(100), np.random.randint(100)))
#         data.append(img)
#
#     return data


# ================================
# for file_name in os.listdir(os.path.join(ANC_PATH)): # 1st step
# for file_name in os.listdir(os.path.join(POS_PATH)): # 2nd step
#     img_path = os.path.join(POS_PATH, file_name)
#     img = cv2.imread(img_path)
#     augmented_images = data_aug(img)
#
#     for image in augmented_images:
#         cv2.imwrite(os.path.join(POS_PATH, f"{uuid.uuid1()}.jpg"), image.numpy())


# ======================================

# ========get images from directories====step-4======
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)

dir_test = anchor.as_numpy_iterator()
print('step-1')


# ---------scale and resize images--------
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0

    return img


# =========create label sets====step-5=====
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()
print('step-2')


# =======train and test partitions=====step-5=====
def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


# ------build dataloader pipeline-------
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=2048)

# ------training partition-----------
train_data = data.take(round(len(data) * .7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# ------testing partition-----------
test_data = data.skip(round(len(data) * .7))
test_data = test_data.take(round(len(data) * .3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# ========build embedding layer====step-6====
def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # ----first layer------
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # ----second layer------
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # ----third layer------
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # ----final layer------
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()
print('step-3')


# ========build distance layer====step-6====
class L1Dist(Layer):
    def __int__(self, **kwargs):
        super().__init__()

    # -----similarity calculating--------
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# ---------make siamese model------
def make_siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # ------combine siamese distance components------
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # -----classification layer--------
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()

# ========setup losses and optimizer========
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

# ------create training checkpoints folder------
checkpoint_dir = './training_checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# =====
# test_batch = train_data.as_numpy_iterator()
# batch_1 = test_batch.next()
# X = batch_1[:2]
# y = batch_1[2]
# =====
print('step-4')


# ------train setup function------
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]

        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)

    # calculate gradient
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # calculate weight and apply siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


# -------training loop-------
def train(data, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print(f"Epochs {epoch}/{EPOCHS}")
        progbar = tf.keras.utils.Progbar(len(data))
        # ================================
        r = Recall()
        p = Precision()

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx + 1)

            print(loss.numpy(), r.result().numpy(), p.result().numpy())
        # ================================
        # save checkpoint
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# =========evaluate models and calculate metric====step-7======
Precision = tf.keras.metrics.Precision
Recall = tf.keras.metrics.Recall

# ------get batch of test data---------
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
# ------make some predictions-------
y_hat = siamese_model.predict([test_input, test_val])
# ------post processing results-------
[1 if prediction > 0.5 else 0 for prediction in y_hat]
# ----^--- or ---^---
# res = []
# for prediction in y_hat:
#     if prediction > 0.5:
#         res.append(1)
#     else:
#         res.append(0)

# -----create metric object and calculate recall values------
# ================================
m = Recall()
m.update_state(y_true, y_hat)
m.result().numpy()
# ----------
m = Precision()
m.update_state(y_true, y_hat)
m.result().numpy()

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true, yhat)

# ================================

# --------virtualize results------
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(test_input[1])
plt.subplot(1, 2, 2)
plt.imshow(test_val[1])
plt.show()

# ======save models=====step-8=======
# save weight
# siamese_model.save('siamesemodel.h5')
siamese_model.save('siamesemodelv2.h5')
# Reload model
model = tf.keras.models.load_model('siamesemodelv2.h5',
                                   custom_objects={'L1Dist': L1Dist,
                                                   'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
# make prediction with reload model
model.predict([test_input, test_val])

# ======real time test=====step-=======
application_data_path = 'application_data'
if not os.path.isdir(application_data_path):
    os.makedirs(application_data_path)
    if not os.path.isdir(f"{application_data_path}/input_image"):
        os.makedirs(f"{application_data_path}/input_image")
    if not os.path.isdir(f"{application_data_path}/verification_images"):
        os.makedirs(f"{application_data_path}/verification_images")


# verification function ========
def verify(model, detection_threshold, verification_threshold):
    # detect input image and varify it====
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # make prediction
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection threshold: make prediction is positive======
    detection = np.sum(np.array(results) > detection_threshold)
    print(detection)

    # Verification threshold: positive prediction / total positive sample======
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold
    print(verified)

    return results, verified


print('step-5')

# =======real time testing========
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]

    # verification ====
    if cv2.waitKey(1) & 0xFF == ord('v'):
        # save input image----
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)

        # verification----
        detection_threshold = 0.8
        verification_threshold = 0.5
        results, verified = verify(model, detection_threshold, verification_threshold)
        print(verified)

    cv2.imshow("Verification", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
