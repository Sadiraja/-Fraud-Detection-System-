import pandas as pd
import numpy as np
import random
from collections import Counter
import math

# Step 1: Data Preprocessing
def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Prevent division by zero
    return (X - mean) / std

def smote(X, y, minority_class=1, k=3, oversample_ratio=0.3):
    print("Applying SMOTE...")
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y != minority_class)[0]
    X_minority = X[minority_indices]
    n_samples = int(len(majority_indices) * oversample_ratio) - len(minority_indices)
    
    if n_samples <= 0:
        print("No synthetic samples needed.")
        return X, y
    
    def get_nearest_neighbors(X, idx, k):
        distances = np.sum((X - X[idx])**2, axis=1)
        distances[idx] = np.inf
        return np.argsort(distances)[:k]
    
    synthetic_X, synthetic_y = [], []
    for i in range(n_samples):
        idx = random.choice(minority_indices)
        minority_idx = np.where(minority_indices == idx)[0][0]
        nn_indices = get_nearest_neighbors(X_minority, minority_idx, k)
        nn = random.choice(nn_indices)
        alpha = random.random()
        synthetic_sample = X_minority[minority_idx] + alpha * (X_minority[nn] - X_minority[minority_idx])
        synthetic_X.append(synthetic_sample)
        synthetic_y.append(minority_class)
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{n_samples} synthetic samples")
    
    X = np.vstack([X, np.array(synthetic_X)])
    y = np.concatenate([y, np.array(synthetic_y)])
    print("SMOTE completed.")
    return X, y

# Step 2: Custom Decision Tree
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini_impurity(self, y):
        counts = Counter(y)
        impurity = 1 - sum((count / len(y))**2 for count in counts.values())
        return impurity

    def find_best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature, best_threshold = None, None
        current_impurity = self.gini_impurity(y)
        
        for feature in feature_indices:
            values = X[:, feature]
            # Sample a subset of thresholds to speed up
            thresholds = np.percentile(values, np.linspace(0, 100, 20))  # 20 quantiles
            for threshold in thresholds:
                left_indices = np.where(values <= threshold)[0]
                right_indices = np.where(values > threshold)[0]
                
                if len(left_indices) < self.min_samples_split or len(right_indices) < self.min_samples_split:
                    continue
                
                left_impurity = self.gini_impurity(y[left_indices])
                right_impurity = self.gini_impurity(y[right_indices])
                weighted_impurity = (len(left_indices) * left_impurity + len(right_indices) * right_impurity) / len(y)
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if len(y) < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        n_features = X.shape[1]
        feature_indices = random.sample(range(n_features), int(np.sqrt(n_features)))
        feature, threshold = self.find_best_split(X, y, feature_indices)
        
        if feature is None:
            return {'class': Counter(y).most_common(1)[0][0]}
        
        left_indices = np.where(X[:, feature] <= threshold)[0]
        right_indices = np.where(X[:, feature] > threshold)[0]
        
        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {'feature': feature, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_single(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        return self.predict_single(x, node['right'])

    def predict(self, X):
        return [self.predict_single(x, self.tree) for x in X]

# Step 3: Custom Random Forest
class RandomForest:
    def __init__(self, n_trees=5, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            print(f"Training tree {i + 1}/{self.n_trees}")
            tree = DecisionTree(self.max_depth, self.min_samples_split)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        print("Random Forest training completed.")

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return [Counter(pred).most_common(1)[0][0] for pred in predictions.T]

# Step 4: Evaluation Metrics
def evaluate_model(true_labels, pred_labels):
    tp = tn = fp = fn = 0
    for true, pred in zip(true_labels, pred_labels):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Step 5: Command-Line Testing Interface
def test_transaction(classifier, feature_names, means, stds):
    print("\nEnter transaction details (or type 'exit' to quit):")
    while True:
        inputs = []
        for feature in feature_names:
            value = input(f"Enter {feature} (float): ")
            if value.lower() == 'exit':
                return
            try:
                inputs.append(float(value))
            except ValueError:
                print("Invalid input. Please enter a number.")
                return
        
        inputs = np.array([inputs])
        inputs = (inputs - means) / stds
        prediction = classifier.predict(inputs)[0]
        print(f"Prediction: {'Fraudulent' if prediction == 1 else 'Non-Fraudulent'}")

# Main function
def main():
    # Load dataset and subsample
    print("Loading dataset...")
    df = pd.read_csv('creditcard.csv')  # Update with actual path
    df = df.sample(n=50000, random_state=42)  # Subsample to 50,000 rows
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    print(f"Dataset loaded: {len(X)} samples")

    # Standardize features
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1
    X = (X - means) / stds

    # Apply SMOTE
    X, y = smote(X, y, minority_class=1, k=3, oversample_ratio=0.3)

    # Split data (80% train, 20% test)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Train Random Forest
    classifier = RandomForest(n_trees=5, max_depth=5, min_samples_split=2)
    classifier.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    predictions = classifier.predict(X_test)
    precision, recall, f1 = evaluate_model(y_test, predictions)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Command-line testing interface
    feature_names = df.drop('Class', axis=1).columns.tolist()
    test_transaction(classifier, feature_names, means, stds)

if __name__ == "__main__":
    main()