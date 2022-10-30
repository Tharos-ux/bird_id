import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def modeling(data_directory: str, batch_size: int, img_height: int, img_width: int, training_steps: int):
    # assuming graphs are saved in directories, grouped by species name
    data_dir: Path = Path(f"{data_directory}/")

    # building the train dataset
    train_ds: list = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # building the validation dataset
    val_ds: list = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # get all the different classes names (here, folders)
    class_names: list[str] = train_ds.class_names

    # buffering to increase performance (à modif)
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # building model
    model: tf.keras.models.Sequential = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(
            1./255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # couche de convolution = nécessaire pour traiter des images en DL
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names))  # sortie
    ])

    print(model.summary())

    # model compilation
    # choose optimizer between SGD, ...
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy']
                  )

    # train model (for training_steps iterations)
    model_training_inforations = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_steps
    )

    # tracer les loss functions au cours des itérations permet de montrer l'overfit si on a divergence au-delà d'un point
    return model, class_names


def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def save_model(trained_model):
    tf.keras.models.save_model(
        model=trained_model, filepath=f"/model_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")


def prediction(entry_path: str, trained_model: tf.keras.models.Sequential, img_height, img_width, class_names) -> str:
    """Does a prediction from a img file

    Args:
        entry_path (str): path to file to test
        trained_model (tf.keras.models.Sequential): a trained model

    Returns:
        str: a string giving the name of the most probable bird
    """
    img = tf.keras.utils.load_img(
        entry_path, target_size=(img_height, img_width))  # penser à la normaliser hihi
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = trained_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return f"This bird sound most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence."
