import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = r"C:/Users/jsy27/OneDrive/바탕 화면/미니인턴 장수열/lumbar_spinal_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "training")
TEST_DIR = os.path.join(BASE_DIR, "testing")

IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 8
VAL_SPLIT = 0.15
SEED = 42

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\n[클래스 인덱스]", train_gen.class_indices)

def build_model(img_size, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

num_classes = train_gen.num_classes
model = build_model(IMG_SIZE, num_classes)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc_ovr")]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

results = model.evaluate(test_gen)
print("\n[Test results]", dict(zip(model.metrics_names, results)))

y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
