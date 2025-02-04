# Topic: Machine learning
## Machine Learning type: Supervised learning
### Supervised learning framework: scikit-learn

Machine learning types:
- supervised
- unsupervised
- reinforcement learning

Supervised learning types:
- regression
- classification
- clustering

Frameworks used:
- scikit-learn


Other frameworks:
- tensorflow
- pytorch
- xgboost
- lightgbm
- catboost

## Basic sci-kit-learn supervised learning workflow
``` python
from sklearn.module import Module

# create model
model = Module()

# train model
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

```

## Supervised learning: Classification
### Classification algorithm: k-nearest neighbors

Building a k-NN classifier from customer churn (cancelling subscriptions, etc) dataset.
Predicting if a customer will cancel a subscription (1) or not (0).

``` python

# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)


# here is new datapoints
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# Predict the labels for the new data
y_pred = knn.predict(X_new)

# Print the predictions
print("Predictions: {}".format(y_pred))

# output is an array
# [0 1 0]
# This means the first and 3rd customers won't cancel their subscription.  But how do we know how accurate this prediction is?  Need to measure model performance.

```
### Measuring classification model performance

- $$accuracy = \frac{correct\ predictions}{total\ observations}$$
- split data into training and test sets.  calculate accuracy on test

![](/zz-img/20250202_13-51-01.png)
![](/zz-img/20250202_13-51-53.png)

![](/zz-img/20250202_14-00-50.png)