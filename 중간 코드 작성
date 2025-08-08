import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def my_vgg_model(img_size=(224,224,1), label_num=5, l2_val=0.0, drop_rate=0.5):
    regul = tf.keras.regularizers.l2(l2_val) if l2_val > 0 else None
    inp = layers.Input(shape=img_size)
    def make_block(x, f):
        x = layers.Conv2D(f, 3, activation='relu', padding='same', kernel_regularizer=regul)(x)
        x = layers.Conv2D(f, 3, activation='relu', padding='same', kernel_regularizer=regul)(x)
        x = layers.MaxPooling2D(2)(x)
        return x
    x = make_block(inp, 64)
    x = make_block(x, 128)
    x = make_block(x, 256)
    x = make_block(x, 512)
    x = make_block(x, 512)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu', kernel_regularizer=regul)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(4096, activation='relu', kernel_regularizer=regul)(x)
    x = layers.Dropout(drop_rate)(x)
    out_layer = layers.Dense(label_num, activation='sigmoid')(x)
    model_obj = models.Model(inp, out_layer)
    return model_obj

def setup_model(m, lr_val=1e-4, loss_fn='binary_crossentropy'):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr_val),
        loss=loss_fn,
        metrics=[tf.keras.metrics.AUC(name='auroc', multi_label=True),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return m

# ===== 실제 데이터 들어오면 이 부분 수정 =====
# def make_dummy(n=256, img_size=(224,224,1), label_num=5, seed_val=42):
#     rng = np.random.default_rng(seed_val)
#     X = rng.normal(size=(n,)+img_size).astype('float32')
#     y = rng.integers(0, 2, size=(n, label_num)).astype('float32')
#     return X, y
#
# def train_dummy_once(m, epochs_val=3, batch_val=16):
#     X, y = make_dummy()
#     hist = m.fit(X, y, validation_split=0.2, epochs=epochs_val, batch_size=batch_val, verbose=0)
#     X_te, y_te = make_dummy(n=64, seed_val=777)
#     evals = m.evaluate(X_te, y_te, verbose=0)
#     result_dict = dict(zip(m.metrics_names, evals))
#     return hist, result_dict
#
# def train_real_data(m, X_train, y_train, X_test, y_test, epochs_val=3, batch_val=16):
#     hist = m.fit(X_train, y_train, validation_split=0.2, epochs=epochs_val, batch_size=batch_val, verbose=1)
#     evals = m.evaluate(X_test, y_test, verbose=1)
#     result_dict = dict(zip(m.metrics_names, evals))
#     return hist, result_dict
