PATH_DATA: str = "data"
PATH_TRAIN: str = "train_filtered_30"
PATH_TRAIN_FILTER: str = "train_filtered"
PATH_UNK: str = "unknown"
MAX_SPECTRO: int = 500  # max de spectro par espèce
HEIGHT: int = 500  # nb de pixels en hauteur
WIDTH: int = 400  # nb de pixels en largeur
ITERATIONS: int = 1  # number of time we do the models

MODEL_PARAMS_1: dict = {
    'model_name': '256_filtered_6L_30sp',
    'epochs': 100,
    'early_stopping': True,
    'batch': 256,
    'validation_split': 0.2,
    'layer_01_filter_count': 8,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_05_filter_count': 32,
    'layer_06_filter_count': 16,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_05_kernel_size': 3,
    'layer_06_kernel_size': 3,
    'layer_dense_size': 64,
    'dropout': 0.2,
    'num_layers': 6,
    'l1_regularization': 0.0,
    'l2_regularization': 0.01
}

MODEL_PARAMS_2: dict = {
    'model_name': 'MODEL_Moana_64',
    'epochs': 100,
    'early_stopping': True,
    'batch': 64,
    'validation_split': 0.2,
    'layer_01_filter_count': 8,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_3: dict = {
    'model_name': 'MODEL_Moana_128',
    'epochs': 100,
    'early_stopping': True,
    'batch': 128,
    'validation_split': 0.2,
    'layer_01_filter_count': 8,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_4: dict = {
    'model_name': 'MODEL_Moana_256',
    'epochs': 100,
    'early_stopping': True,
    'batch': 256,
    'validation_split': 0.2,
    'layer_01_filter_count': 8,
    'layer_02_filter_count': 8,
    'layer_03_filter_count': 16,
    'layer_04_filter_count': 32,
    'layer_01_kernel_size': 5,
    'layer_02_kernel_size': 3,
    'layer_03_kernel_size': 3,
    'layer_04_kernel_size': 3,
    'layer_dense_size': 32,
    'dropout': 0.0,
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.0
}

MODEL_PARAMS_5: dict = {
    'model_name': 'MODEL_L2_32',
    'epochs': 100,
    'early_stopping': True,
    'batch': 32,
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
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.1
}

MODEL_PARAMS_6: dict = {
    'model_name': 'MODEL_L2_64',
    'epochs': 100,
    'early_stopping': True,
    'batch': 64,
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
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.1
}

MODEL_PARAMS_7: dict = {
    'model_name': 'MODEL_L2_128',
    'epochs': 100,
    'early_stopping': True,
    'batch': 128,
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
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.1
}

MODEL_PARAMS_8: dict = {
    'model_name': 'MODEL_L2_256',
    'epochs': 256,
    'early_stopping': True,
    'batch': 32,
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
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.1
}

MODEL_PARAMS_9: dict = {
    'model_name': 'FULL_256',
    'epochs': 5,
    'early_stopping': True,
    'batch': 256,
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
    'num_layers': 4,
    'l1_regularization': 0.0,
    'l2_regularization': 0.1
}

LIST_OF_MODELS: list = [MODEL_PARAMS_1]
