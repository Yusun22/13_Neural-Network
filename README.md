# 13_Neural-Network

### This project aims to create a model that predicts whether businesses will become successful based on a variety of information give about each business. This information will be used as features to create a binary classifier model using a deep neural network that will predict whether the business applicant will become a successful if funding is received.

---

## Technologies

This project leverages python 3.9 and [Google Colab](https://colab.research.google.com/?utm_source=scs-index) was used to run all analysis.

---

## Installations

Before running the application first install and import the following libraries and dependencies.

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```

---

## Data Preparation

The dataset contained both categorical and numerical variables. In order to analyze them, we encoded the categorical variables so they are transformed into numerical values, specifically into binary classification.

`OneHotEncoder` was imported and it was used to numerically encode all of the dataset's categorical data, where then the new dataset was saved into a new DataFrame. Below is the code that creates the `OneHotEncoder` instance:

```python
enc = OneHotEncoder(sparse=False)
```

Next, features (x) and target (y) were created. The target was set to `IS_SUCCESSFUL` column and the features were set to all other columns. `StandardScaler` was used to scale the split data.

---

## Creating a Neural Network Model

Below are the codes used for creating a deep neural network where the number of input features, layers, and neurons on each layer were assigned when using [Tensorflow's Keras](https://www.tensorflow.org/api_docs/python/tf/keras).

```python
number_input_features = 116
hidden_nodes_layer1 = 58
hidden_nodes_layer2 = 29

nn = Sequential()

nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))

nn.add(Dense(units=1, activation="sigmoid"))
```

Then, we compile and then fit our deep neural network model. The following code compiles and fits the model:

```python
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

fit_model = nn.fit(X_train_scaled, y_train, epochs=50)
```

Finally, the model was evaluated and saved to an HDF5 file.

---

## Optimizing the Neural Network Model

Two more alternative models were built in order to improve on the first model's predictive accuracy. Below is the summary of the results:

- **Original Model**:
  268/268 - 0s - loss: 0.5530 - accuracy: 0.7294 - 292ms/epoch - 1ms/step
  Loss: 0.5530143976211548, Accuracy: 0.7294460535049438

* **Alternative Model 1**: `epochs=100` was used
  268/268 - 0s - loss: 0.5595 - accuracy: 0.7300 - 373ms/epoch - 1ms/step
  Loss: 0.5594684481620789, Accuracy: 0.7300291657447815

- **Alternative Model 2**: a second hidden layer was added, `epochs=100`, `activation="leaky_relu"` and `batch_size=32` were used instead
  268/268 - 0s - loss: 0.5652 - accuracy: 0.7305 - 298ms/epoch - 1ms/step
  Loss: 0.5651640892028809, Accuracy: 0.7304956316947937

**Conclusion**: Model 2 had the highest accuracy score and therefore, this model would be recommended for use in predicting a business' success.
