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

## split data into training and test sets
``` python
# Import the module
from sklearn.model_selection import train_test_split

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
```

## Determine how many neighbors to use to prevent over/underfitting
``` python
# Create neighbors
neighbors = np.arange(1,13)
train_accuracies = {}
test_accuracies = {}

# determine how many neighbors we should use to prevent overfitting and underfitting.  Loop through each of the neighbor values 1:13
for neighbor in neighbors:
  
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)
  
	# Fit the model
	knn.fit(X_train, y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)
```

## Visualize how performance changes as model complexity changes.  
``` python
# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()
```

See how training accuracy decreases and test accuracy increases as the number of neighbors gets larger. For the test set, accuracy peaks with 7 neighbors, suggesting it is the optimal value for our model. Now let's explore regression models!

![](/zz-img/20250203_17-50-23.png)

---

# Intro to regression

Goal: predict blood glucose levels
![](/zz-img/20250203_17-51-41.png)

