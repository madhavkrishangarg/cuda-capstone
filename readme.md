# Deep Voice Deepfake Voice Recognition

## Overview

This repository contains a project focused on detecting deepfake voices using convolutional neural network analysing spectrograms of voices. The project leverages a dataset from Kaggle to train and evaluate models capable of distinguishing between real and synthetic voices.

## Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition). It contains audio samples labeled as either real or deepfake. The dataset is structured to facilitate training and testing of voice recognition models.

## Libraries and Tools

The project utilizes several Python libraries and tools, including:

- **Librosa**: For audio processing and feature extraction. It is used to load audio files and extract features such as Mel-frequency cepstral coefficients (MFCCs).
- **NumPy**: For numerical operations and handling arrays.
- **Pandas**: For data manipulation and analysis, particularly for handling metadata associated with the audio files.
- **Scikit-learn**: For building and evaluating machine learning models. It provides tools for splitting the dataset, training models, and assessing their performance.
- **TensorFlow/Keras**: For building deep learning models. These libraries are used to create and train neural networks for voice recognition.
- **Matplotlib/Seaborn**: For data visualization. These libraries are used to plot data distributions, model performance metrics, and other visualizations.
- **PyCUDA**: For leveraging GPU acceleration to speed up computationally intensive tasks. It allows for the execution of custom CUDA kernels and facilitates parallel processing, which can be particularly beneficial for training deep learning models or performing large-scale audio data processing tasks efficiently.

## Project Structure

- `data/`: Contains the audio dataset and any additional data files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.