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
from time import process_time, monotonic
from itertools import chain


def plot_metrics(metrics, classes_names: list, predictions: list, labels: list, path_to_save: str = None):
    """Plots out metrics and test set results

    Args:
        metrics (history): model fitting informations
        classes_names (list): all classes used for train
        predictions (list): predicted results for test set instances
        labels (list): true results for test set instances
        path_to_save (str, optional): path where model will be stored. Defaults to None.
    """

    fig, axs = plt.subplots(figsize=(16, 9), dpi=100, ncols=2, nrows=2)
    axs[0, 0].title.set_text('Fig. A : Accuracy')
    axs[0, 1].title.set_text('Fig. B : Loss')
    axs[1, 0].title.set_text('Fig. C : Validation set accuracy')
    axs[1, 1].title.set_text('Fig. D : Validation set loss')

    x = [i+1 for i in range(len(metrics.history['accuracy']))]

    axs[0, 0].plot(x, metrics.history['accuracy'])
    axs[0, 1].plot(x, metrics.history['loss'])
    axs[1, 0].plot(x, metrics.history['val_accuracy'])
    axs[1, 1].plot(x, metrics.history['val_loss'])

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

    # clustering matrix
    rcParams['figure.figsize'] = 25, 20
    sns.clustermap(cm, cmap=sns.cubehelix_palette(as_cmap=True),
                   cbar_pos=None, xticklabels=True, yticklabels=True, annot=False)
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/cluster_matrix.png", transparent=True)
    else:
        plt.show()

    plt.close()

    # TODO annot=True
    # confusion matrix
    rcParams['figure.figsize'] = 25, 20
    ax = plt.axes()
    ax.set_title(f"Confusion matrix")
    sns.heatmap(cm, annot=False, cmap=sns.cubehelix_palette(
        as_cmap=True), linewidths=0.5, ax=ax)
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/conf_matrix.png", transparent=True)
    else:
        plt.show()

    plt.close()


def resnet_model(params, class_names, include_top=False, weights=None, input_shape=None, layer_params=[2, 2, 2, 2], pooling=None):
    BN_AXIS = 3
    DATA_FORMAT = 'channels_last'

    def ResNet18(params, class_names, include_top=False, weights=None, input_shape=None, layer_params=[2, 2, 2, 2], pooling=None):
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=32,
                                          data_format=DATA_FORMAT,
                                          require_flatten=include_top,
                                          weights=weights)

        img_input = tf.keras.layers.Input(shape=input_shape)

        x = tf.keras.layers.ZeroPadding2D(
            padding=(3, 3), name='conv1_pad')(img_input)
        x = tf.keras.layers.Conv2D(64, (7, 7),
                                   strides=(2, 2),
                                   padding='valid',
                                   kernel_initializer='he_normal',
                                   name='conv1')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=BN_AXIS, name='bn_conv1')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = make_basic_block_layer(x, filter_num=64,
                                   blocks=layer_params[0])
        x = make_basic_block_layer(x, filter_num=128,
                                   blocks=layer_params[1],
                                   stride=2)
        x = make_basic_block_layer(x, filter_num=256,
                                   blocks=layer_params[2],
                                   stride=2)
        x = make_basic_block_layer(x, filter_num=512,
                                   blocks=layer_params[3],
                                   stride=2)

        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

        x = tf.keras.layers.Flatten()(x)

        model = tf.keras.Model(img_input, x, name='resnet18')
        return model

    def make_basic_block_base(inputs, filter_num, stride=1):
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   kernel_initializer='he_normal',
                                   padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)
        x = tf.keras.layers.Conv2D(filters=filter_num,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   kernel_initializer='he_normal',
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=BN_AXIS)(x)

        shortcut = inputs
        if stride != 1:
            shortcut = tf.keras.layers.Conv2D(filters=filter_num,
                                              kernel_size=(1, 1),
                                              strides=stride,
                                              kernel_initializer='he_normal')(inputs)
            shortcut = tf.keras.layers.BatchNormalization(
                axis=BN_AXIS)(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def make_basic_block_layer(inputs, filter_num, blocks, stride=1):
        x = make_basic_block_base(inputs, filter_num, stride=stride)

        for _ in range(1, blocks):
            x = make_basic_block_base(x, filter_num, stride=1)

        return x

    def _obtain_input_shape(input_shape,
                            default_size,
                            min_size,
                            data_format,
                            require_flatten,
                            weights=None):
        """
        Private function taken from Tensorflow internal library.
        """
        if weights != 'imagenet' and input_shape and len(input_shape) == 3:
            if data_format == 'channels_first':
                default_shape = (input_shape[0], default_size, default_size)
            else:
                default_shape = (default_size, default_size, input_shape[-1])
        else:
            if data_format == 'channels_first':
                default_shape = (3, default_size, default_size)
            else:
                default_shape = (default_size, default_size, 3)
        if weights == 'imagenet' and require_flatten:
            if input_shape is not None:
                if input_shape != default_shape:
                    raise ValueError('When setting `include_top=True` '
                                     'and loading `imagenet` weights, '
                                     '`input_shape` should be ' +
                                     str(default_shape) + '.')
            return default_shape
        if input_shape:
            if data_format == 'channels_first':
                if input_shape is not None:
                    if len(input_shape) != 3:
                        raise ValueError(
                            '`input_shape` must be a tuple of three integers.')
                    if input_shape[0] != 3 and weights == 'imagenet':
                        raise ValueError('The input must have 3 channels; got '
                                         '`input_shape=' + str(input_shape) + '`')
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         '; got `input_shape=' +
                                         str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if len(input_shape) != 3:
                        raise ValueError(
                            '`input_shape` must be a tuple of three integers.')
                    if input_shape[-1] != 3 and weights == 'imagenet':
                        raise ValueError('The input must have 3 channels; got '
                                         '`input_shape=' + str(input_shape) + '`')
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         '; got `input_shape=' +
                                         str(input_shape) + '`')
        else:
            if require_flatten:
                input_shape = default_shape
            else:
                if data_format == 'channels_first':
                    input_shape = (3, None, None)
                else:
                    input_shape = (None, None, 3)
        if require_flatten:
            if None in input_shape:
                raise ValueError('If `include_top` is True, '
                                 'you should specify a static `input_shape`. '
                                 'Got `input_shape=' + str(input_shape) + '`')
        return input_shape
    return ResNet18(params, class_names, include_top, weights, input_shape, layer_params, pooling)


def naive_model(img_height: int, img_width: int, params: dict, class_names: list):
    """Inits a CNN-style model (keras-sequential) from given parameters

    Args:
        img_height (int): height for images
        img_width (int): width for images
        params (dict): dict of params, as defined in 'constants.py'
        class_names (list): all classes used for train

    Returns:
        Sequential: descriptions of layers as an object
    """

    return tf.keras.models.Sequential(
        list(
            chain(
                *[
                    [
                        tf.keras.layers.Rescaling(
                            1./255, input_shape=(img_height, img_width, 3))
                    ]
                    +
                    list(
                        chain(
                            *[
                                [
                                    tf.keras.layers.Conv2D(
                                        params[f'layer_0{i+1}_filter_count'], params[f'layer_0{i+1}_kernel_size'], padding='same', activation='relu')
                                ]
                                +
                                [
                                    tf.keras.layers.MaxPooling2D()
                                ] for i in range(params['num_layers'])
                            ]
                        )
                    )
                    +
                    [
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dropout(
                            params['dropout']),
                        tf.keras.layers.Dense(
                            params['layer_dense_size'], activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(l1=params['l1_regularization'], l2=params['l2_regularization'])),
                        tf.keras.layers.Dense(
                            len(class_names), activation='softmax')  # sortie
                    ]
                ]
            )
        )
    )


def modeling(data_directory: str, img_height: int, img_width: int, params: dict, save_status: bool, resnet: bool, save_path="models") -> tuple:
    """Calls for model creation and fitting, and then evaluates metrics for this model with a test set.

    Args:
        data_directory (str): path to train
        img_height (int): size of images
        img_width (int): size of images
        params (dict): dict of params, as defined in 'constants.py'
        save_status (bool): tells if model should be saved to drive when computation ends
        resnet (bool): tells if model should be resnet
        save_path (str, optional): output path for saving model. Defaults to "models".

    Returns:
        tuple: (model, classes names)
    """
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
        batch_size=params['batch'],
        shuffle=True
    )

    # building the validation dataset
    val_ds: list = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=params['validation_split'],
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=params['batch'],
        shuffle=True
    )

    # get all the different classes names (here, folders)
    class_names: list[str] = train_ds.class_names
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # building model
    if resnet:
        model = resnet_model(params, class_names, include_top=False, weights=None,
                             input_shape=None, layer_params=[2, 2, 2, 2], pooling=None)
    else:
        model = naive_model(img_height, img_width, params, class_names)

    model.summary()

    # model compilation
    # choose optimizer between SGD, ...
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy']
                  )

    start_time = monotonic()
    start_time_cpu = process_time()
    # train model (for training_steps iterations)
    if params['early_stopping']:
        model_training_informations = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params['epochs'],
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=3, restore_best_weights=True)]
        )
    else:
        model_training_informations = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params['epochs']
        )
    end_time = monotonic()
    end_time_cpu = process_time()

    # get predictions
    predictions, labels = [], []
    for x, y in val_ds:
        predictions = np.concatenate(
            [predictions, np.argmax(model.predict(x), axis=-1)])
        labels = np.concatenate([labels, y.numpy()])

    m = tf.keras.metrics.Accuracy()
    print(f"Validation accuracy = {m(labels, predictions).numpy()}")
    # compute confusion matrix with `tf`
    confusion = tf.math.confusion_matrix(
        labels=labels,   # get true labels
        predictions=predictions  # get predicted labels
    ).numpy()

    print(confusion)

    # tracer les loss functions au cours des itérations permet de montrer l'overfit si on a divergence au-delà d'un point
    save_model(model, class_names, model_training_informations,
               predictions, labels, save_status, params, end_time_cpu - start_time_cpu, end_time - start_time, save_path)
    print(
        f"Finished model computation, ended after {len(model_training_informations.history['loss'])} epochs.")
    return model, class_names


def load_model(model_path: str) -> tuple:
    """Loads a model and a list of classes from a previously saved model

    Args:
        model_path (str): path to folder containing the saved files

    Returns:
        tuple: (model,list of classes)
    """
    return tf.keras.models.load_model(model_path), load(open(f"{model_path}/classes.json", "r"))


def save_model(trained_model, classes: list, model_training_informations, predictions: list, labels: list, save_status: bool, params: dict, cpu_exec_time: int, exec_time: int, save_path: str) -> None:
    """Saves a trained model to disk for later use

    Args:
        trained_model (tf.model): trained model
        classes (list): list of classes used for train
        model_training_informations (history): history object containing iterations informations
        predictions (list): list of predictions for test set
        labels (list): true labels for test set
        save_status (bool): if should save to drive
        params (dict): model dict parameters (constants.py format)
        cpu_exec_time (int): processor time of fitting the model
        exec_time (int): real time for model fitting
        save_path (str): path where model will be stored
    """
    out_path = None
    if save_status:
        out_path: str = f"{save_path}/{params['model_name']}_{params['iter']}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
        tf.keras.models.save_model(
            model=trained_model, filepath=out_path)
        dump(classes, open(f"{out_path}/classes.json", "w"))
        dump({'execution_time': exec_time, 'cpu_time': cpu_exec_time, 'true_epochs': len(
            model_training_informations.history['accuracy']), 'accuracy_train': model_training_informations.history['accuracy'], 'loss_train': model_training_informations.history['loss'], 'accuracy_validation': model_training_informations.history['val_accuracy'], 'loss_validation': model_training_informations.history['val_loss'], **params}, open(f"{out_path}/params.json", "w"))
        with open(f"{out_path}/model.txt", "w") as writer:
            trained_model.summary(print_fn=lambda x: writer.write(x + '\n'))
    plot_metrics(model_training_informations, classes,
                 predictions, labels, out_path)


def prediction(entry_path: str, trained_model: tf.keras.models.Sequential, img_height, img_width, class_names) -> str:
    """Does a prediction from a img file

    Args:
        entry_path (str): path to file to test
        trained_model (tf.keras.models.Sequential): a trained model
        img_height (int): size of img
        img_width (int): size of img
        class_names (list): list of classes used for train

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
