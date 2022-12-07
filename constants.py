PATH_DATA: str = "data"
PATH_TRAIN: str = "train_unfiltered_sampled"
PATH_TRAIN_FILTER: str = "train_filtered"
PATH_UNK: str = "unknown"
MAX_SPECTRO: int = 500  # max de spectro par espèce
HEIGHT: int = 500  # nb de pixels en hauteur
WIDTH: int = 400  # nb de pixels en largeur


MODEL_PARAMS_1: dict = {
    # nombre de packs d'itérations d'entrainements (epochs = k itérations du gradient)
    'epochs': 10,
    'early_stopping': False,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.2,
    'layer_01_filter_count': 4,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 16,
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

LIST_OF_MODELS:list = [MODEL_PARAMS_1]