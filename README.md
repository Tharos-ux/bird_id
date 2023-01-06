# Birdcall identification

## Overview

This project is about the usage of deep learning for classification, using spectrograms which encodes .mp3 files.
The software is composed of a full pipeline, covering from data preprocessing to results interpretation, with one minimal example to showcase its usage.
A small overview of the key functions of the project is available in the Jupyter Notebook `overview_project.ipynb`.

## Usage

You may use the main.py file to interact with the software, and the constants.py file for tweaking parameters.

```bash
usage: main.py [-h] [-c] [-s] [-p] [-m MODEL] [-o] [-f] [-r]

options:
  -h, --help            show this help message and exit
  -c, --clean           Erase all data inside training folder
  -s, --spectrograms    Builds the spectrograms from the current data folder
  -p, --predict         Predicts unknown spectrograms from a train folder   
  -m MODEL, --model MODEL
                        Path to a saved model
  -o, --output          Model will be saved on disk for later use
  -f, --filter          Specifies spectro filtering
  -r, --resnet          Uses ResNet architecture for model
```

## Implemented functions

```
NAME
    tensorflow_model

FUNCTIONS
    load_model(model_path: str) -> tuple
        Loads a model and a list of classes from a previously saved model

        Args:
            model_path (str): path to folder containing the saved files

        Returns:
            tuple: (model,list of classes)

    modeling(data_directory: str, img_height: int, img_width: int, params: dict, save_status: bool, resnet: bool, save_path='models') -> tuple
        Calls for model creation and fitting, and then evaluates metrics for this model with a test set.

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


    naive_model(img_height: int, img_width: int, params: dict, class_names: list)
        Inits a CNN-style model (keras-sequential) from given parameters

        Args:
            img_height (int): height for images
            img_width (int): width for images
            params (dict): dict of params, as defined in 'constants.py'
            class_names (list): all classes used for train

        Returns:
            Sequential: descriptions of layers as an object

    plot_metrics(metrics, classes_names: list, predictions: list, labels: list, path_to_save: str = None)
        Plots out metrics and test set results

        Args:
            metrics (history): model fitting informations
            classes_names (list): all classes used for train
            predictions (list): predicted results for test set instances
            labels (list): true results for test set instances
            path_to_save (str, optional): path where model will be stored. Defaults to None.

    prediction(entry_path: str, trained_model: keras.engine.sequential.Sequential, img_height, img_width, class_names) -> str
        Does a prediction from a img file

        Args:
            entry_path (str): path to file to test
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

NAME
    metadata_extract

FUNCTIONS
    extract_name(input: str, output: str = 'spec_name.json')
        Extract specie name infos from metadata .csv into .json
        Args:
            input (str): metadata file
            output (str, optional): file path to store name infos. Defaults to "spec_name.json".

    extract_rating(input: str, output: str = 'rating.json')
        Extract rating infos from metadata .csv into .json
        Args:
            input (str): metadata file
            output (str, optional): file path to store rating infos. Defaults to "rating.json".

NAME
    audio_sampling

FUNCTIONS
    audio_processing(data_path: str, output_path: str, specie: str, max_spectro: int = 700, rating_max: float = 4, filter: bool = False) -> None
        Exports raw audios into pre-processed spectrograms

        Args:
            data_path (str): directory containing species folders
            output_path (str): output master directory
            specie (str): name of subfolder
            max_spectro (int, optional): limits the number of spectrograms to plot per specie. Defaults to 700.
            rating_max (float, optional): defines a target score level for audio, refering to a 'rating.json' file. Defaults to 4.
            filter (bool, optional): tells if a restrictive filter should be applied to chunks. Defaults to False.

    export_spectro(l_chunks: list, specie_name: str, filename: str, output_path: str)
        Converts audio into spectros and exports them
            /!\ SPECTROS ARE 500x400px for consistency issues --> some weren't this size without fixed params

        Args:
            l_chunks (list): list of all audio chunks to plot
          the file descriptor must refer to a directory.                                                                                                                                                                            
          If this functionality is unavailable, using it raises NotImplementedError.                                                                                                                                                
                                                                                                                                                                                                                                    
        The list is in arbitrary order.  It does not include the special                                                                                                                                                            
        entries '.' and '..' even if they are present in the directory.                                                                                                                                                             
                                                                                                                                                                                                                                    
    load_in_blocks(audio_path: str, frame_size: int = 3, limit_chunks: int = 100, filter: bool = False, overlap: float = 0.5)                                                                                                       
        Chunks audio into parts of 'frame_size' seconds                                                                                                                                                                             
                                                                                                                                                                                                                                    
        Args:                                                                                                                                                                                                                       
            entry_path (str): path to audio
            frame_size (int, optional): chunks size. Defaults to 5.
            overlap (float, optional): overlap percentage
        Returns:
            list of chunks

NAME
    main

FUNCTIONS
    futures_collector(func: Callable, argslist: list, num_processes: int) -> list
        Spawns len(arglist) instances of func and executes them at num_processes instances at time.
        
        * func : a function
        * argslist (list): a list of tuples, arguments of each func
        * num_processes (int) : max number of concurrent instances
```
