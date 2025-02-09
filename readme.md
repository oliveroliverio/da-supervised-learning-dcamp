
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

sklearn requires target and feature variables/data in distinct variables, 
- x = features = all other columns: `X = diabetes_df.drop("glucose", axis=1).values`
- y = target = glucose values: `y = diabetes_df["glucose"].values`

![](/zz-img/20250203_17-58-02.png)

### Making predictions from a single feature/column. 

``` python
# get only BMI feature
X_bmi = X[:, 3]
# check to see what dimensional arrays these are.  
print(y.shape, X_bmi.shape)

# since these are 1-dimensional arrays, we need to convert the features 2D for sklearn
# do this by reshaping
X_bmi = X_bmi.reshape(-1, 1)
print(X_bmi.shape)

# (752, 1) is the correct shape now for model 
```

### Plot glucose vs. bmi

```python
import matplotlib.pyplot as plt
plt.scatter(X_bmi, y)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("BMI")
plt.show() 
```
![](/zz-img/20250203_18-04-13.png)

## Fit regression model to data
use linear regression to fit straight line to data
``` python
from sklearn.linear_model import LinearRegression
# create model
reg = LinearRegression()
# train model
reg.fit(X_bmi, y)
# predict by fitting a line
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions, color="red")
plt.xlabel("BMI")
plt.ylabel("Blood Glucose (mg/dl)")
plt.show()
```
![](/zz-img/20250203_18-07-29.png)

## Practice Regression! 

