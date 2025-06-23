import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score,roc_auc_score, roc_curve, precision_score, confusion_matrix, classification_report
import pandas as pd
import rasterio as r
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingSet():
    """
    Class for handling Local Climate Zone (LCZ) training and testing datasets.

    This class manages the preparation, scaling, and splitting of predictor and target arrays
    for LCZ classification tasks.

    Args:
        X_image (np.ndarray): Predictor image stack (n_features, height, width).
        y_train_image (np.ndarray): Training target image(s).
        y_test_image (np.ndarray): Testing target image(s).
        X_names (list): List of predictor (feature) names.
        y_labels (dict): Mapping of class values to class names.
        rescale (bool): Whether to apply MinMax scaling to predictors.

    Attributes:
        X_image (np.ndarray): Scaled or original predictor image stack.
        y_train_image (np.ndarray): Training target image(s).
        y_test_image (np.ndarray): Testing target image(s).
        X_names (list): Feature names.
        y_labels (dict): Class value to name mapping.
        y_names (list): List of class names.
        y_classes (list): List of class values.
        n_classes (int): Number of classes.
        y_train (np.ndarray): Flattened, masked training targets.
        y_test (np.ndarray): Flattened, masked testing targets.
        X_train (np.ndarray): Masked training predictors.
        X_test (np.ndarray): Masked testing predictors.

    Methods:
        print_dimensions(): Print dataset dimensions and info.
        rescale(): Apply MinMax scaling to predictors.
        split_samples(set, test_size, stratify, random_state): Split data into train/test sets.
    """

    def __init__(self, X_image, y_train_image, y_test_image, X_names, y_labels, encode = False, rescale = True):
        
        # Setup X Image
        self.X_image = X_image
        if rescale:
            # Initialize Input Parameters
             self.rescale()


        self.y_train_image = y_train_image
        self.y_test_image = y_test_image
        self.X_names = X_names
        self.y_labels = y_labels
        self.y_names = list(y_labels.values())
        self.y_classes = list(y_labels.keys())
        self.n_classes = len(self.y_classes)
        self.encoder_train = LabelEncoder()
        self.encoder_test = LabelEncoder()
        # Format Training Datset
        y_train = self.y_train_image[0].reshape(-1)
        y_train_mask = y_train > 0
        
        self.y_train = y_train[y_train_mask]

        # Format Testing Datset
        y_test = self.y_test_image[0].reshape(-1)
        y_test_mask = y_test > 0
        self.y_test = y_test[y_test_mask]

        if encode:
            self.y_train = self.encoder_train.fit_transform(self.y_train)
            self.y_test =  self.encoder_test.fit_transform(self.y_test)

        
        # Build Predictor Stack for Input Images
        X = predictor_stack(self.X_image)
        self.X_train = X[y_train_mask] # Mask predictor stack with target class mask from training set
        self.X_test = X[y_test_mask] # Mask predictor stack with target class mask from testing set
        


    def print_dimensions(self):
        print('TRAINING SET DIMENSIONS')
        print('==============================')
        
        print(f'X Image: {self.X_image.shape}')
        print(f'X Predictors: {self.X_train.shape}')
        print(f'X Band Names: {self.X_names}')
        print('')
        print(f'y Image: {self.y_train_image.shape}')
        print(f'y Classes: {self.y_train.shape}')
        print(f'y Class Count: {self.n_classes}')
        print(f'y Class Names: {self.y_names}')
        print('==============================')

    def rescale(self):

        scaler = MinMaxScaler()
        self.X_image = np.array([scaler.fit_transform(x.reshape(-1,1)).reshape(x.shape) for x in self.X_image])

    def split_samples(self, set ='train', test_size = 0.2, stratify = False, random_state = 42):

        X,y = (self.X_train, self.y_train) if set == 'train' else (self.X_test, self.y_test)
        # Pass y array if stratify == True
        if stratify:
            stratify_classes=y
        else:
            stratify_classes= None

        
        # Split dataset based on test size and stratify
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_classes, random_state=random_state)

        print('TRAIN TEST SPLIT')
        print('==============================')
        
        print(f'X Train: {self.X_train.shape}')
        print(f'y Train: {self.y_train.shape}')
        print(f'X Test: {self.X_test.shape}')
        print(f'y Test: {self.y_test.shape}')

        return X_train, X_test, y_train, y_test

    def get_samples(self):
        return self.X_train, self.X_test, self.y_train, self.y_test



def predictor_stack(features):
    """
    Build a predictor stack from a 3D array with shape (n_features, height, width).

    Args:
        features (np.array): 3D array with shape (n_features, height, width)

    Returns:
        np.array: Predictor stack with shape (height*width, n_features)
    """
    return np.stack(features).reshape(features.shape[0],-1).T




def search_parameters(classification_method, X_train, y_train, search_method='random', scoring='accuracy', cv=5, verbose=10, param_grid = None):
    """
    Perform hyperparameter tuning using GridSearchCV for a given classifier.

    Args:
        classification_method (str): Key for classifier in classifiers dict.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        scoring (str): Scoring metric for GridSearchCV.
        cv (int): Number of cross-validation folds.
        verbose (int): Verbosity level for GridSearchCV.

    Returns:
        GridSearchCV: Fitted GridSearchCV object with best parameters.
    """
    
    cl_info = classifiers.get(classification_method)
    cl, name = cl_info.get('cl'), cl_info.get('name')
    
    if param_grid is None:
        param_grid = cl_info.get('param_grid')
    # if classification_method == 'RandomForest':
    #     cl = cl(oob_score=True)
    # else:
    cl = cl()

    
    print(f'Performing {search_method.title()} Search with Classifier: {name}')
    print(f'====================================================================')
    print(f'Parameter Grid:')
    for key in param_grid.keys():
        print(f'{key}: {param_grid[key]}')
    print(f'====================================================================')

    search_class = searchers.get(search_method)

    kwargs = dict(
            estimator=cl, 
            scoring=scoring, 
            cv=cv, 
            verbose=verbose,
            return_train_score =True
        )
    
    

    if search_method =='random':
        kwargs.update(dict(param_distributions = param_grid))
    else:
        kwargs.update(dict(param_grid = param_grid))
    # Create a GridSearchCV object to find the best hyperparameters
    search = search_class(**kwargs)
    

    # Fit the GridSearchCV object to the training data
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print("Best hyperparameters:", best_params)
    
    return search



def compute_metrics(y_true, y_pred, y_proba, class_names):
    """
    Compute various classification metrics for model evaluation.

    This function calculates multiple performance metrics including accuracy,
    F1-score, recall, precision, ROC AUC, confusion matrix, and a detailed classification
    report for multi-class classification tasks.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        y_proba (array-like): Predicted class probabilities.
        class_names (list): List of class names.

    Returns:
        dict: Dictionary containing the following metrics:
            - accuracy (float): Overall classification accuracy
            - f1_score (float): Weighted F1-score
            - recall_score (float): Weighted recall score
            - precision_score (float): Weighted precision score
            - roc_auc_score (float): Weighted ROC AUC score (OvR)
            - confusion (array): Confusion matrix
            - report (dict): Detailed classification report with per-class statistics
    """
    return dict(
        accuracy = accuracy_score(y_true, y_pred),
        f1_score = f1_score(y_true, y_pred, average='weighted'),
        recall_score = recall_score(y_true, y_pred, average='weighted'),
        precision_score = precision_score(y_true, y_pred, average='weighted'),
        # roc_auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'),
        confusion = confusion_matrix(y_true, y_pred),
        report = classification_report(y_true, y_pred,  target_names= class_names),
        report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True,  target_names= class_names)),
    )


def print_comparison_report(report1, report2, color_threshold=0.8):
    """
    Print a compact colored comparison of classification metrics between two models.
    Colors indicate performance levels:
        RED: < 0.8 (poor)
        YELLOW: 0.8-0.9 (good)
        GREEN: >= 0.9 (excellent)

    Args:
        report1 (pd.DataFrame): Classification report for model 1.
        report2 (pd.DataFrame): Classification report for model 2.
        color_threshold (float): Threshold for coloring performance.

    Returns:
        None
    """
    cl_report = report1.join(report2, rsuffix='_pr')
    
    # Format support columns as integers
    cl_report['support'] = cl_report['support'].astype(int)
    cl_report['support_pr'] = cl_report['support_pr'].astype(int)
    
    # Define color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    def get_color(val):
        try:
            val_float = float(val)
            if val_float >= 0.9:
                return f"{GREEN}{val:.2f}{END}"
            if val_float >= 0.8 and   val_float < 0.9:
                return f"{YELLOW}{val:.2f}{END}"
            else:
                return f"{RED}{val:.2f}{END}"
        except:
            return val
    
    # Format each metric with colors
    formatted_df = cl_report.copy()
    metric_cols = ['precision', 'recall', 'f1-score', 
                  'precision_pr', 'recall_pr', 'f1-score_pr']
    
    for col in metric_cols:
        formatted_df[col] = formatted_df[col].apply(get_color)
    
    # Shorter headers without newlines
    headers = [
        'Class', 'Pre-S2', 'Rec-S2', 'F1-S2', 'N-S2',
        'Pre-PR', 'Rec-PR', 'F1-PR', 'N-PR'
    ]
    
    print(f"\n{BOLD}LCZ Classification Metrics - S2 vs PRISMA{END}")
    print(tabulate(
        formatted_df, 
        headers=headers,
        floatfmt=".2f",  # Reduced decimal places
        tablefmt="simple",  # Changed from "grid" to "simple" for more compact output
        numalign="right",
        stralign="left",
        colalign=("left",) + ("right",) * 8  # Align numbers right, class names left
    ))


def print_metrics_comparison(metrics1, metrics2, model1_name="Model 1", model2_name="Model 2"):
    """
    Print a side-by-side comparison of key metrics for two models with difference.
    Colors indicate performance levels:
        RED: < 0.8 (poor)
        YELLOW: 0.8-0.9 (good)
        GREEN: >= 0.9 (excellent)

    Args:
        metrics1 (dict): Dictionary containing metrics from first model
        metrics2 (dict): Dictionary containing metrics from second model
        model1_name (str): Name of first model for display
        model2_name (str): Name of second model for display

    Returns:
        None
    """
    # Define color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    def get_color(val):
        if val >= 0.9:
            return GREEN
        elif val >= 0.8:
            return YELLOW
        return RED
    
    # Create comparison data
    metrics = ['accuracy', 'f1_score', 'recall_score', 'precision_score', 'roc_auc_score']
    display_names = ['Accuracy', 'F1 Score', 'Recall', 'Precision', 'AUC Score']
    
    # Format data for tabulate
    rows = []
    for metric, display_name in zip(metrics, display_names):
        val1, val2 = metrics1[metric], metrics2[metric]
        diff = val2 - val1  # Calculate difference
        
        # Apply color based on thresholds
        val1_str = f"{get_color(val1)}{val1:.2f}{END}"
        val2_str = f"{get_color(val2)}{val2:.2f}{END}"
        
        # Color the difference
        if val1 > val2:
            diff_str = f"{RED}{diff:+.2%}{END}"
        elif val2 > val1:
            diff_str = f"{GREEN}{diff:+.2%}{END}"
        else:
            diff_str = f"{diff:+.2%}"
            
        rows.append([display_name, val1_str, val2_str, diff_str])
    
    # Print comparison table
    print(f"\n{BOLD}Model Performance Comparison{END}")
    print(tabulate(
        rows,
        headers=[f"Metric", model1_name, model2_name, "Performance Change"],
        tablefmt="simple",
        numalign="right",
        stralign="right"
    ))



"""
Dictionary of supported classifiers and their parameter grids for hyperparameter tuning.

Keys:
    RandomForest, AdaBoost, GradientBoost, XGBoost

Each value is a dict with:
    cl (sklearn/base.BaseEstimator): Classifier instance
    param_grid (dict): Parameter grid for GridSearchCV
    name (str): Human-readable classifier name
"""

classifiers = dict(
  
    RandomForest = dict(
        cl = RandomForestClassifier,
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'criterion': ['gini', 'entropy'],
            # 'max_depth': [2, 5, 10],
            # 'min_samples_split': [100,200,300],
       
            # 'min_samples_leaf': randint(1, 20),
            # 'max_leaf_nodes': [20, 50, 100, 200, 500, 1000], # Include None as an option
            # 'min_impurity_decrease': [0.001, 0.005, 0.01, 0.03, 0.05], # Small values, can be more specific like [0.0, 0.001, 0.005]
       
            # 'min_samples_leaf': [25, 50],
            'class_weight': ['balanced','balanced_subsample', None]
        },
        name = 'Random Forest Classifier'
    ),
    AdaBoost = dict(
        cl = AdaBoostClassifier,
        param_grid = {
            'n_estimators': [100, 150, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        name = 'AdaBoost Classifier'
    ),
    GradientBoost = dict(
        cl = GradientBoostingClassifier,
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_features': ['log2', 'sqrt', 'log2'],
            'criterion': ['friedman_mse', 'mae', 'mse'],
            'n_estimators': [100, 150, 200, 500]
        },
        name = 'Gradient Boosting Classifier'
    ),
    # XGBoost = dict(
    #     cl = xgb.XGBClassifier,
    #     param_grid = {
    #         'n_estimators': [100, 150, 200, 500],
    #         'learning_rate': [0.01, 0.05, 0.1, 0.2]
    #     },
    #     name = 'XGBoost Classifier'
    # )
)


searchers = dict(
    random = RandomizedSearchCV,
    grid = GridSearchCV
)



def print_classifier_params_table(clf1, clf2, clf1_name="Classifier 1", clf2_name="Classifier 2"):
    """
    Print a side-by-side table of parameters for two classifiers.

    Args:
        clf1: First classifier (must have get_params method)
        clf2: Second classifier (must have get_params method)
        clf1_name (str): Name for first classifier column
        clf2_name (str): Name for second classifier column
    """
    params1 = clf1.get_params()
    params2 = clf2.get_params()
    all_keys = sorted(set(params1.keys()).union(params2.keys()))

    data = []
    for key in all_keys:
        val1 = params1.get(key, "")
        val2 = params2.get(key, "")
        data.append([key, val1, val2])

    df = pd.DataFrame(data, columns=["Parameter", clf1_name, clf2_name])
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
