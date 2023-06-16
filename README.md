# Venture Funding with Deep Learning

## Overview
This project focuses on predicting the success of venture funding for startups using deep learning techniques. The goal is to build accurate binary classification models that can assist in determining whether applicants will be successful if funded by Alphabet Soup, a venture capital firm.

The project includes the following steps:

1. Data Preparation: The provided dataset, "applicants_data.csv," is preprocessed using Pandas and scikit-learn's StandardScaler to prepare it for the neural network model. Categorical variables are encoded, and numerical variables are scaled.

2. Binary Classification Model: A deep neural network model is designed using TensorFlow and Keras. The model utilizes the dataset's features to predict the success of Alphabet Soup-funded startups. The model is compiled, fitted, and evaluated to calculate the loss and accuracy.

3. Model Optimization: Multiple attempts are made to optimize the model's accuracy. Different techniques are employed, such as adjusting input data, adding more neurons or hidden layers, using different activation functions, and modifying the number of training epochs. Each optimized model is evaluated and compared to the original model.

## Project Structure

The project is organized into the following components:

- `Colab_venture_funding_with_deep_learning.ipynb`: Google Colab notebook file containing the code for data preprocessing, model training, evaluation, and optimization.

- `Resources`: Directory containing the dataset.
- `applicants_data.csv`: The dataset inside the `Resources` directory containing information about the startups that have received funding from Alphabet Soup.

- `Saved_Models`: Directory containing all 4 saved models.
- `AlphabetSoup.h5`: The original saved model in HDF5 format.
- `AlphabetSoupA1.h5`: First additional optimized model saved in HDF5 format.
- `AlphabetSoupA2.h5`: Second additional optimized model saved in HDF5 format.
- `AlphabetSoupA3.h5`: Third additional optimized model saved in HDF5 format.

- `README.md`: The README file providing an overview of the project.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- Pandas
- TensorFlow
- Keras
- scikit-learn

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/hiddenciphers/13-venture_funding_with_deep_learning.git`
2. Navigate to the project directory: `cd 13-venture_funding_with_deep_learning`
3. Upload the notebook file to Google Colab: `Colab_venture_funding_with_deep_learning.ipynb`
4. Run the cells in the Notebook to preprocess the data, train the model, evaluate its performance, optimize & save the models.

## Model Results

### Original Model Results
- Loss: 0.5524504780769348
- Accuracy: 0.7308454513549805

### Alternative Model 1 Results
- Loss: 0.5512396097183228
- Accuracy: 0.7307288646697998

### Alternative Model 2 Results
- Loss: 0.5539257526397705
- Accuracy: 0.7290962338447571

### Alternative Model 3 Results
- Loss: 0.5540934205055237
- Accuracy: 0.7303789854049683


## Conclusion

The Venture Funding with Deep Learning project demonstrates the application of deep learning techniques to predict the success of venture funding for startups. The models built in this project can assist Alphabet Soup in making informed decisions about funding applications, increasing the efficiency of resource allocation.

For more details on the project and the models, please refer to the Google Colab Notebook and the saved model files.

