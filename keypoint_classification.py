import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import streamlit as st
import time

RANDOM_SEED = 42
dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'


def getGestureLabel():
    with open(label_path,
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    return keypoint_classifier_labels


def clearTfModel():
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite'
    ):
        interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=1)
        self.interpreter.allocate_tensors()
        interpreter.keras.backend.clear_session()


def run_keypoint_classification():
    clearTfModel()
    global NUM_CLASSES
    NUM_CLASSES = len(getGestureLabel()) + 1
    with st.spinner('Processing....'):
        databaseReading()
        modelBuilding()
        inferenceTest()
        time.sleep(0)
        st.title('complete')


def databaseReading():
    X_dataset = np.loadtxt(dataset, delimiter=',',
                           dtype='float32', usecols=list(range(1, (21 * 2) + 1)))

    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)


def modelBuilding():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    # Model compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )

    # Model evaluation
    val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
    st.markdown(f"value loss : {val_loss}")
    st.markdown(f"value accurancy : {val_acc}")

    # Loading the saved model
    model = tf.keras.models.load_model(model_save_path)
    # Inference test
    predict_result = model.predict(np.array([X_test[0]]))
    print(np.squeeze(predict_result))
    print(np.argmax(np.squeeze(predict_result)))
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    # confusion matrix
    confusionMatrix(y_test, y_pred)

    # Save as a model dedicated to inference
    model.save(model_save_path, include_optimizer=False)
    # Transform model (quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    open(tflite_save_path, 'wb').write(tflite_quantized_model)


def confusionMatrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    # plt.show()
    st.pyplot(fig)
    st.title('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)


def inferenceTest():
    interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
    interpreter.allocate_tensors()
    # Get I / O tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
    # Inference implementation
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    print(np.squeeze(tflite_results))
    print(np.argmax(np.squeeze(tflite_results)))
