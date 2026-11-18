# GTZAN-Classification

**A convolutional neural network that classifies 30-second music clips from the GTZAN dataset into ten genres, working on mel spectrograms of 4-second audio chunks.**

> **Context.** Built in April 2025 as a personal machine learning project, when I was a Computer Engineering student at UPV and my public GitHub was just starting. It is one of my first repositories: a single notebook, trained and run locally on Windows. It is kept as it was written; the code is not maintained.

## What it does

- Loads the GTZAN audio dataset (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock).
- Splits each track into 4-second chunks with 2 seconds of overlap and turns each chunk into a mel spectrogram resized to 150x150.
- Trains a 5-block CNN on those spectrograms and saves it as `Trained_model.h5`.
- Evaluates it with accuracy, a per-genre classification report and a confusion matrix, and plots the training curves.
- Predicts the genre of a single audio file (the first 4 seconds, padded if shorter).

## How it works

The notebook `music_Classification.ipynb` follows these steps:

1. Visualize one waveform and its mel spectrogram, whole and chunked.
2. Build the dataset: every `.wav` file is cut into overlapping 4-second chunks, each converted with `librosa.feature.melspectrogram` and resized with `skimage` to 150x150x1. This gives 14,975 samples, split 80/20 into 11,980 for training and 2,995 for testing (`train_test_split`, `random_state=42`).
3. Model (Keras `Sequential`, 7,182,458 trainable parameters): five blocks of two `Conv2D` layers (3x3, ReLU) plus `MaxPool2D`, with 32, 64, 128, 256 and 512 filters. Dropout of 0.3 after the third and fifth blocks, then `Flatten`, `Dense(1200)`, `Dropout(0.45)` and a 10-unit softmax.
4. Training: Adam with learning rate 0.0001, categorical crossentropy, 20 epochs, batch size 32, using the test split as validation data. The history is stored in `training_hist.json`.
5. Evaluation and inference on the saved model.

One limitation visible in the code: the split is random over chunks, so overlapping chunks of the same track can end up in both the training and the test set. The accuracy below should be read with that in mind.

## Results

All figures are printed in the notebook outputs (section 7 and 9), measured on the 2,995 test chunks:

- Training accuracy 98.4 percent (loss 0.056), validation accuracy 89.1 percent (loss 0.404). The notebook itself comments that this gap suggests some overfitting.
- Per genre F1 from the classification report: classical 0.96, hiphop 0.95, metal 0.95, jazz 0.91, disco 0.90, reggae 0.90, blues 0.86, pop 0.85, country 0.83, rock 0.80. Macro average 0.89.
- Rock is the weakest genre (recall 0.74), most often confused with country (23 of 291 test chunks), then blues and metal (14 each), per the confusion matrix.

## Run it

Not run again for this README; the commands below are inferred from the notebook imports.

1. Download the GTZAN dataset from Kaggle (see `readme.txt`): https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Install the dependencies: `pip install tensorflow librosa matplotlib numpy scikit-image scikit-learn seaborn jupyter`
3. Open `music_Classification.ipynb` and change the hard-coded Windows paths (`C:\Users\<user>\Downloads\GTZAN\...`) to your dataset location and to where the model should be saved.
4. Run the cells in order. Building the dataset and training 20 epochs took roughly 4 minutes per epoch on the author's machine (see the epoch logs); a GPU is advisable.
5. Note that `Trained_model.h5` is not in the repository, so section 7 only works after running the training cells (or after training once and saving the file).

## Repository layout

```
music_Classification.ipynb   full pipeline: data, model, training, evaluation, prediction
training_hist.json           loss and accuracy per epoch (train and validation)
readme.txt                   link to the GTZAN dataset, which is too big for GitHub
```

## Stack

Python, TensorFlow and Keras, librosa, scikit-image, scikit-learn, NumPy, Matplotlib, Seaborn, Jupyter.
