# Spaceship Titanic – End-to-End ML Pipeline

A machine learning project for the Kaggle Spaceship Titanic competition.
The project contains two complementary notebooks:

1. An end-to-end machine learning workflow from exploratory data analysis to deployment.
2. A production-oriented scikit-learn pipeline with modular preprocessing and a soft-voting ensemble.

## Results

| Metric          |                                                                     Result |
| --------------- | -------------------------------------------------------------------------: |
| Kaggle rank     |                                                                        423 |
| Kaggle accuracy |                                                                     80.64% |
| Competition     | [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) |

The competition uses classification accuracy as its evaluation metric.
The task is to predict whether a passenger was transported to another dimension.

## Project Overview

The Spaceship Titanic competition provides passenger records from a fictional spaceship accident.
The goal is to predict the binary target variable `Transported`.

The project focuses on two different aspects of machine learning:

- Understanding the complete machine learning workflow.
- Building a reusable and production-oriented preprocessing and prediction pipeline.

## Notebooks

### 1. End-to-End Machine Learning Workflow (spaceship-titanic)

This notebook covers the complete machine learning lifecycle:

- Loading and inspecting the data
- Exploratory data analysis
- Data cleaning
- Missing-value analysis
- Feature engineering
- Feature selection
- Model training
- Model comparison
- Validation and error analysis
- Final model training
- Prediction and submission generation

### 2. Scikit-Learn Pipeline (spaceship-titanic-pipeline)

The second notebook focuses on creating a reusable machine learning pipeline.

The preprocessing workflow consists of:

1. A feature-generation pipeline for creating derived features.
2. A `ColumnTransformer` for numerical and categorical preprocessing.
3. Imputation and categorical encoding.
4. A postprocessing step for the final feature transformations.
5. A model ensemble for generating predictions.

Keeping preprocessing and model training inside a single pipeline helps ensure that the same transformations are applied consistently during training and inference.

## Feature Engineering

The main challenge of the dataset is that several features are related to each other.
Therefore, feature engineering was used to transform the raw passenger information into more useful model features.

Examples include:

- Extracting information from the `Cabin` feature.
- Separating cabin information into deck, number, and side.
- Aggregating spending-related features.
- Creating group- or passenger-related features.
- Handling missing values according to feature type.

## Model Architecture

The final submission was generated using a soft-voting ensemble consisting of:

- `XGBClassifier`
- `LogisticRegression`
- `LGBMClassifier`
- `SVC(kernel="rbf", probability=True)`

The ensemble uses the following weights:

```text
XGBClassifier:       2
LogisticRegression:  1
LGBMClassifier:      2
SVC:                 1
```

The soft-voting classifier combines the predicted class probabilities of the individual models.
This allows the ensemble to use complementary behavior from tree-based, linear, and kernel-based classifiers.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.11+
```

## Dataset

The dataset is provided by the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic).

The original dataset contains passenger information such as:

- Passenger group
- Home planet
- CryoSleep status
- Cabin
- Destination
- Age
- VIP status
- Expenditure-related features

The target variable is:

```text
Transported
```

The target indicates whether a passenger was transported to another dimension.

## Limitations

- The Kaggle score depends on the selected validation strategy and feature engineering decisions.
- Some engineered features are specific to the Spaceship Titanic dataset.
- The pipeline is production-oriented but is not yet exposed through a standalone inference API.
- The notebooks still contain experimentation and are not a complete packaged Python application.
