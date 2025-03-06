# Deep Learning-Based Spam Detection for Email Classification

## Overview

This repository contains a deep learning model designed to classify email messages into spam and non-spam categories. The project leverages TensorFlow and Keras to apply natural language processing (NLP) and deep learning techniques to a dataset of email messages.

## Repository Structure

- **`DL_spam_shield.ipynb`**: The Jupyter notebook that contains the entire machine learning pipeline, from data loading and preprocessing to model training and evaluation.
- **`spam.csv`**: The dataset containing labeled email messages used for training the model, obtained from [Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).
- **`spam_classifier_model.h5`**: The saved Keras model that encapsulates the trained neural network, allowing for immediate use without retraining.

## Machine Learning Model

The integration of NLP and deep learning in this project involves several sophisticated steps:

### Data Preprocessing
- **Text Cleaning**: Removes email headers, normalizes text to lower case, and eliminates punctuation.
- **Vectorization**: Uses TF-IDF to convert cleaned text into numerical vectors that effectively represent word frequencies while minimizing the impact of frequently occurring words that might be less informative.

### Model Architecture
- **Deep Neural Network**: Comprises layers specifically structured for binary classification:
  - **Input Layer**: Accepts TF-IDF vectorized data.
  - **Hidden Layers**: Include multiple dense layers with ReLU activation and dropout for regularization to prevent overfitting.
  - **Output Layer**: Uses a sigmoid activation function to output a probability indicating the likelihood of spam.

### Libraries and Frameworks
- **TensorFlow & Keras**: For building and training the neural network model.
- **Scikit-learn**: For data splitting and model evaluation.
- **NLTK**: For natural language processing, including stopwords removal.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Matplotlib & Seaborn**: For visualizing training results and performance metrics.

### Hyperparameter Tuning
- **Process**: Utilizes Keras Tuner to systematically explore a range of configurations, optimizing model hyperparameters such as layer sizes, dropout rates, and learning rates to achieve the best validation performance.
- **Outcome**: Selection of the optimal model configuration based on the highest area under the curve (AUC) metric on the validation set.

### Training and Evaluation
- **Early Stopping**: Monitors validation loss, halting training when improvement ceases, thereby preventing overfitting.
- **Performance Metrics**: Evaluates the model using accuracy, precision, recall, and AUC to ensure comprehensive assessment of model effectiveness.

## Usage Instructions

### Setup
Install the necessary Python packages to ensure the notebook runs smoothly:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn nltk sklearn wordcloud
```

### Running the Notebook
- Open `DL_spam_shield.ipynb` in Jupyter Notebook.
- Execute all cells to see the process from data loading to model training and evaluation.

## Acknowledgements
This project is a practical application of deep learning and NLP, demonstrating the capabilities of modern AI technologies in tackling real-world problems like spam detection in emails.
