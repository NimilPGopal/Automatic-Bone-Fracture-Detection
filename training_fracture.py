import numpy as np
import pandas as pd
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from tensorflow.keras.optimizers import Adam
from training_parts import LEARNING_RATE
from core.preprocessing import clahe_preprocessing


def load_path(path, part):
    """
    load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                if body == part:
                    body_part = body
                    path_p = folder + '/' + str(body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = path_p + '/' + str(id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            else:
                                continue
                            path_l = path_id + '/' + str(lab)
                            for img in os.listdir(path_l):
                                img_path = path_l + '/' + str(img)
                                dataset.append(
                                    {
                                        'body_part': body_part,
                                        'patient_id': patient_id,
                                        'label': label,
                                        'image_path': img_path
                                    }
                                )
    return dataset


# this function get part and know what kind of part to train, save model and save plots
def trainPart(part):
    # categories = ['fractured', 'normal']
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = THIS_FOLDER + '/Dataset/'
    data = load_path(image_dir, part)
    labels = []
    filepaths = []

    # add labels for dataframe for each category 0-fractured, 1- normal
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    # ---------------------------------------------------
    # PATIENT-LEVEL SPLIT
    # ---------------------------------------------------

    patient_df = pd.DataFrame(data)

    unique_patients = patient_df['patient_id'].unique()

    train_patients, test_patients = train_test_split(
        unique_patients,
        train_size=0.9,
        random_state=42,
        shuffle=True
    )

    train_df = patient_df[
        patient_df['patient_id'].isin(train_patients)
    ]

    test_df = patient_df[
        patient_df['patient_id'].isin(test_patients)
    ]

    train_df = train_df.rename(columns={
        'image_path': 'Filepath',
        'label': 'Label'
    })

    test_df = test_df.rename(columns={
        'image_path': 'Filepath',
        'label': 'Label'
    })


    # each generator to process and convert the filepaths into image arrays,
    # and the labels into one-hot encoded labels.
    # The resulting generators can then be used to train and evaluate a deep learning model.



    if part == "Hand":

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=clahe_preprocessing,
            validation_split=0.2,

            rotation_range=5,
            zoom_range=0.04,

            width_shift_range=0.08,
            height_shift_range=0.08,

            brightness_range=[0.90, 1.10],

            horizontal_flip=True,
            fill_mode="nearest"
        )

    elif part == "Shoulder":

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=clahe_preprocessing,
            validation_split=0.2,

            rotation_range=5,
            zoom_range=0.04,

            width_shift_range=0.08,
            height_shift_range=0.08,

            brightness_range=[0.90, 1.10],

            horizontal_flip=True,
            fill_mode="nearest"
        )

    else:

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=clahe_preprocessing,
            validation_split=0.2,

            rotation_range=5,
            zoom_range=0.04,

            width_shift_range=0.08,
            height_shift_range=0.08,

            brightness_range=[0.90, 1.10],

            horizontal_flip=True,
            fill_mode="nearest"
        )






    # use DenseNet121 architecture
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=clahe_preprocessing)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(320,320),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=True,
        seed=42,
        subset='training'
    )

    # ------------------------------------------------------------------
    # Compute class weights to handle class imbalance
    # ------------------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_images.classes),
        y=train_images.classes
    )

    class_weights = dict(enumerate(class_weights))

    print("Class Weights:", class_weights)

    val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=clahe_preprocessing,
        validation_split=0.2
    )

    val_images = val_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(320, 320),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=False,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(320, 320),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=False
    )

    # we use rgb 3 channels and 224x224 pixels images, use feature extracting , and average pooling
    pretrained_model = tf.keras.applications.densenet.DenseNet121(
        input_shape=(320, 320, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    # for faster performance
    # Fine-tuning setup
    for layer in pretrained_model.layers[:-120]:
        layer.trainable = False

    for layer in pretrained_model.layers[-120:]:
        layer.trainable = True

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(pretrained_model.output)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    # outputs Dense '2' because of 2 classes, fratured and normal
    outputs = tf.keras.layers.Dense(
        2,
        activation='softmax'
    )(x)
    model = tf.keras.Model(inputs, outputs)
    # print(model.summary())
    print("-------Training " + part + "-------")

    # Create plot directory automatically
    plot_dir = Path(THIS_FOLDER) / "plots" / "FractureDetection" / part
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Adam optimizer with low learning rate for better accuracy
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        ),
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc'),

            tfa.metrics.F1Score(
                num_classes=2,
                average='macro',
                threshold=0.5,
                name='f1_score'
            )
        ]
    )
    # early stop when our model is over fit or vanishing gradient, with restore best values
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score',
        mode='max',
        patience=7,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_score',
        mode='max',
        factor=0.5,
        patience=5,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=THIS_FOLDER + "/weights/DenseNet121_" + part + "_best.keras",
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=35,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weights
    )

    # save model to this path
    # Load BEST saved model (best val_auc model)
    model = tf.keras.models.load_model(
        THIS_FOLDER + "/weights/DenseNet121_" + part + "_best.keras"
    )    
    
    results = model.evaluate(test_images, verbose=0)
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")
    print(f"Test Recall: {np.round(results[2] * 100, 2)}%")
    print(f"Test Precision: {np.round(results[3] * 100, 2)}%")
    print(f"Test AUC: {np.round(results[4] * 100, 2)}%")

    # create plots for accuracy and save it
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    figAcc.savefig(plot_dir / "Accuracy.jpeg")
    plt.clf()

    # create plots for loss and save it
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    figAcc.savefig(plot_dir / "Loss.jpeg")
    plt.clf()

    # create recall plot and save it
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    figRecall = plt.gcf()
    figRecall.savefig(plot_dir / "Recall.jpeg")
    plt.clf()

    # create AUC plot and save it
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    figAUC = plt.gcf()
    figAUC.savefig(plot_dir / "AUC.jpeg")
    plt.clf()

# run the function and create model for each parts in the array
#categories_parts = ["Elbow", "Hand", "Shoulder"]
#for category in categories_parts:
#    trainPart(category)

trainPart("Shoulder")
