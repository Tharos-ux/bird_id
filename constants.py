PATH_DATA: str = "data"
PATH_TRAIN: str = "train_unfiltered_sampled"
PATH_TRAIN_FILTER: str = "train_filtered"
PATH_UNK: str = "unknown"
MAX_SPECTRO: int = 500  # max de spectro par espèce
HEIGHT: int = 500  # nb de pixels en hauteur
WIDTH: int = 400  # nb de pixels en largeur
ITERATIONS: int = 3  # number of time we do the models

MODEL_PARAMS_1: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_333'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 3,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 3,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_2: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_555'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 5,
    'layer_03_kernel_size': 5,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 3,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_3: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_533'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 3,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_4: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_553'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 5,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 3,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_5: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_535'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 5,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 3,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_6: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_353'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 3,
    'layer_02_kernel_size': 5,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 3,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_7: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_5333'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 4,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_8: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_3555'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 3,
    'layer_02_kernel_size': 5,
    'layer_03_kernel_size': 5,
    'layer_04_kernel_size': 5,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 4,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_9: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_5353'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 5,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 4,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_10: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'model_name': 'MODEL_archi_3535'
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 3,
    'layer_02_kernel_size': 5,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 5,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 4,  # number of layers (supports only 3 and 4)
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

LIST_OF_MODELS: list = [MODEL_PARAMS_1, MODEL_PARAMS_2, MODEL_PARAMS_3, MODEL_PARAMS_4,
                        MODEL_PARAMS_5, MODEL_PARAMS_6]
