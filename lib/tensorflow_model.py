import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
from datetime import datetime
from json import load, dump
from multiprocessing import cpu_count
from math import sqrt
import numpy as np
from pandas import DataFrame, crosstab


def plot_metrics(cm, metrics, training_steps, classes_names, predictions, labels, path_to_save=None):

    fig, axs = plt.subplots(figsize=(16, 9), dpi=100, ncols=2, nrows=2)
    axs[0, 0].title.set_text('Fig. A : Accuracy')
    axs[0, 1].title.set_text('Fig. B : Loss')
    axs[1, 0].title.set_text('Fig. C : Validation set accuracy')
    axs[1, 1].title.set_text('Fig. D : Validation set loss')

    x = [i+1 for i in range(len(metrics.history['accuracy']))]

    axs[0, 0].plot(x, metrics.history['accuracy'])
    axs[0, 1].plot(x, metrics.history['loss'])
    axs[1, 0].plot(x, metrics.history['val_loss'])
    axs[1, 1].plot(x, metrics.history['val_accuracy'])

    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/metrics.png", transparent=True)
    else:
        plt.show()

    plt.close()

    inverted_map = {str(i): class_name for i,
                    class_name in enumerate(classes_names)}

    test_classes = [inverted_map[str(int(t))] for t in labels]
    test_preds = [inverted_map[str(int(t))] for t in predictions]
    data = {'y_Actual': test_classes, 'y_Predicted': test_preds}
    df = DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    cm = crosstab(df['y_Actual'], df['y_Predicted'], rownames=[
        'Actual'], colnames=['Predicted'])
    print('MATRIX : ', cm, 'FIN')

    # clustering
    rcParams['figure.figsize'] = 25, 20
    sns.clustermap(cm, cmap=sns.cubehelix_palette(as_cmap=True),
                   cbar_pos=None, xticklabels=True, yticklabels=True, annot=True)
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/cluster_matrix.png", transparent=True)
    else:
        plt.show()

    plt.close()

    rcParams['figure.figsize'] = 25, 20
    ax = plt.axes()
    ax.set_title(f"Confusion matrix")
    sns.heatmap(cm, annot=True, cmap=sns.cubehelix_palette(
        as_cmap=True), linewidths=0.5, ax=ax)
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/conf_matrix.png", transparent=True)
    else:
        plt.show()

    plt.close()


def modeling(data_directory: str, img_height: int, img_width: int, params: dict, save_status: bool):

    # Allocation of sqrt(threads) cores per process and sqrt(threads) parallel processes
    sqrt_threads: int = int(sqrt(cpu_count()))
    tf.config.threading.set_inter_op_parallelism_threads(sqrt_threads)
    tf.config.threading.set_intra_op_parallelism_threads(sqrt_threads)

    # assuming graphs are saved in directories, grouped by species name
    data_dir: Path = Path(f"{data_directory}/")

    # building the train dataset
    train_ds: list = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=params['validation_split'],
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=params['batch']
    )

    # building the validation dataset
    val_ds: list = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=params['validation_split'],
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=params['batch']
    )

    # get all the different classes names (here, folders)
    class_names: list[str] = train_ds.class_names
    #train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    #val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # building model
    model: tf.keras.models.Sequential = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(
            1./255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(
            params['layer_01_size'], 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            params['layer_02_size'], 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # couche de convolution 2D = nécessaire pour traiter des images en DL
        tf.keras.layers.Conv2D(
            params['layer_03_size'], 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(params['layer_dense_size'], activation='relu'),
        tf.keras.layers.Dense(len(class_names))  # sortie
    ])

    summary = str(model.summary())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    # model compilation
    # choose optimizer between SGD, ...
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy']
                  )

    # train model (for training_steps iterations)
    model_training_informations = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=params['epochs'],
        callbacks=[callback]
    )

    # get predictions

    predictions = np.argmax(model.predict(val_ds), axis=1)
    labels = np.concatenate([y for _, y in val_ds], axis=0)

    # compute confusion matrix with `tf`
    confusion = tf.math.confusion_matrix(
        labels=labels,   # get true labels
        predictions=predictions  # get predicted labels
    ).numpy()

    # tracer les loss functions au cours des itérations permet de montrer l'overfit si on a divergence au-delà d'un point
    save_model(model, class_names, model_training_informations,
               params['epochs'], confusion, predictions, labels, save_status, params, summary)
    print(
        f"Finished model computation, ended after {len(model_training_informations.history['loss'])} epochs.")
    return model, class_names


def load_model(model_path: str):
    return tf.keras.models.load_model(model_path), load(open(f"{model_path}/classes.json", "r"))


def save_model(trained_model, classes, model_training_informations, training_steps, confusion_matrix, predictions, labels, save_status, params, summary):
    out_path = None
    if save_status:
        out_path: str = f"models/model_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
        tf.keras.models.save_model(
            model=trained_model, filepath=out_path)
        dump(classes, open(f"{out_path}/classes.json", "w"))
        dump(params, open(f"{out_path}/params.json", "w"))
        with open(f"{out_path}/model.txt", "w") as writer:
            writer.write(summary)
    plot_metrics(confusion_matrix, model_training_informations,
                 training_steps, classes, predictions, labels, out_path)


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
    with open('names.json', 'r') as file:  # Sortir le nom commun
        names: dict = load(file)
    return [(names[class_names[i]], score[i].numpy()*100) for i in range(len(class_names))]
