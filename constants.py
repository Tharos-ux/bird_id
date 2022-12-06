PATH_DATA: str = "data"
PATH_TRAIN: str = "train_unfiltered_sampled"
PATH_TRAIN_FILTER: str = "train_filtered"
PATH_UNK: str = "unknown"
MAX_SPECTRO: int = 500  # max de spectro par espèce
HEIGHT: int = 500  # nb de pixels en hauteur
WIDTH: int = 400  # nb de pixels en largeur

MODEL_PARAMS: dict = {
    # nombre de packs d'itérations d'entrainements (epocs = k itérations du gradient)
    'epochs': 30,
    'batch': 32,  # taille des paquets de données
    'validation_split': 0.3,
    'layer_01_size': 4,
    'layer_02_size': 8,
    'layer_03_size': 16,
    'layer_dense_size': 32,
}
