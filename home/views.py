from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the dataset
df = pd.read_csv('breast_cancer_data.csv')

# Drop the 'id' column if it exists
if 'id' in df.columns:
    df.drop(['id'], axis=1, inplace=True)
else:
    print("'id' column not found.")

# Shuffle the dataset
np.random.seed(42)  # For reproducibility
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Splitting ratio (80% training, 20% testing)
split_ratio = 0.8
split_index = int(len(df_shuffled) * split_ratio)

# Splitting the dataset into training and testing sets
train_data = df_shuffled.iloc[:split_index]
test_data = df_shuffled.iloc[split_index:]
X_train = train_data.drop(['diagnosis'], axis=1)
y_train = train_data['diagnosis']

X_test = test_data.drop(['diagnosis'], axis=1)
y_test = test_data['diagnosis']
y_train_numeric = y_train.map({'B': 0, 'M': 1}).values
y_test_numeric = y_test.map({'B': 0, 'M': 1}).values

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    return X_normalized, mean, std_dev

X_train_normalized, mean_train, std_dev_train = normalize_features(X_train)
X_train_normalized_bias = np.hstack((np.ones((X_train_normalized.shape[0], 1)), X_train_normalized))

# Normalize the features for testing set using mean and std_dev from training set
X_test_normalized = (X_test - mean_train) / std_dev_train

# Add a bias term to the features for testing set
X_test_normalized_bias = np.hstack((np.ones((X_test_normalized.shape[0], 1)), X_test_normalized))

print("Normalized training set shape:", X_train_normalized_bias.shape, y_train_numeric.shape)
print("Normalized testing set shape:", X_test_normalized_bias.shape, y_test_numeric.shape)

# Initialize parameters (weights and bias)
np.random.seed(42)  # For reproducibility
num_features = X_train_normalized_bias.shape[1]
theta = np.random.randn(num_features)

# Define the sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the learning rate and number of iterations
learning_rate = 0.1  # Adjust the learning rate if needed
num_iterations = 1000

# Perform batch gradient descent with checks for numerical stability
for i in range(num_iterations):
    # Calculate the predicted probabilities
    z = np.dot(X_train_normalized_bias, theta)
    h = sigmoid(z)
    # Calculate the gradient
    gradient = np.dot(X_train_normalized_bias.T, (h - y_train_numeric)) / len(y_train_numeric)

    # Check for numerical stability
    if np.isnan(np.sum(gradient)):
        print("Gradient calculation resulted in NaN values. Adjust learning rate or check data preprocessing.")
        break
    # Update parameters
    theta -= learning_rate * gradient

print("Optimized parameters (theta):\n", theta)

# Calculate predicted probabilities for the test set
z_test = np.dot(X_test_normalized_bias, theta)
h_test = sigmoid(z_test)

# Convert probabilities to binary predictions
y_pred = np.where(h_test >= 0.5, 1, 0)


def detection(request):
    if request.method == "POST":
        
            radius_mean = float(request.POST.get('radius_mean'))
            texture_mean = float(request.POST.get('texture_mean'))
            perimeter_mean = float(request.POST.get('perimeter_mean'))
            area_mean = float(request.POST.get('area_mean'))
            smoothness_mean = float(request.POST.get('smoothness_mean'))
            compactness_mean = float(request.POST.get('compactness_mean'))
            concavity_mean = float(request.POST.get('concavity_mean'))
            concave_points_mean = float(request.POST.get('concave_points_mean'))
            symmetry_mean = float(request.POST.get('symmetry_mean'))
            fractal_dimension_mean = float(request.POST.get('fractal_dimension_mean'))
            radius_se = float(request.POST.get('radius_se'))
            texture_se = float(request.POST.get('texture_se'))
            perimeter_se = float(request.POST.get('perimeter_se'))
            area_se = float(request.POST.get('area_se'))
            smoothness_se = float(request.POST.get('smoothness_se'))
            compactness_se = float(request.POST.get('compactness_se'))
            concavity_se = float(request.POST.get('concavity_se'))
            concave_points_se = float(request.POST.get('concave_points_se'))
            symmetry_se = float(request.POST.get('symmetry_se'))
            fractal_dimension_se = float(request.POST.get('fractal_dimension_se'))
            radius_worst = float(request.POST.get('radius_worst'))
            texture_worst = float(request.POST.get('texture_worst'))
            perimeter_worst = float(request.POST.get('perimeter_worst'))
            area_worst = float(request.POST.get('area_worst'))
            smoothness_worst = float(request.POST.get('smoothness_worst'))
            print(smoothness_worst)
            compactness_worst = float(request.POST.get('compactness_worst'))
            concavity_worst = float(request.POST.get('concavity_worst'))
            concave_points_worst = float(request.POST.get('concave_points_worst'))
            symmetry_worst = float(request.POST.get('symmetry_worst'))
            fractal_dimension_worst = float(request.POST.get('fractal_dimension_worst'))

            input_data = np.array([radius_mean, texture_mean, perimeter_mean, area_mean,
                                   smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
                                   symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se,
                                   area_se, smoothness_se, compactness_se, concavity_se, concave_points_se,
                                   symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                                   perimeter_worst, area_worst, smoothness_worst, compactness_worst,
                                   concavity_worst, concave_points_worst, symmetry_worst,
                                   fractal_dimension_worst]).reshape(1, -1)

            # Normalize the input data using mean_train and std_dev_train from the training set
            X_input_normalized = (input_data - mean_train.values) / std_dev_train.values

            # Add bias term to the normalized input
            X_input_normalized_bias = np.hstack((np.ones((X_input_normalized.shape[0], 1)), X_input_normalized))

            # Use the trained logistic regression model to predict diagnosis
            z_input = np.dot(X_input_normalized_bias, theta)
            h_input = sigmoid(z_input)
            pred = 'M' if h_input >= 0.5 else 'B'
            print(pred)

            return render(request, 'prediction.html', {'pred': pred})
    else:
        return render(request, 'detection.html')  # Render the empty form initially
       

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html/')
