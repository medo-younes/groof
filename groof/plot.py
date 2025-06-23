# Standard library imports
import warnings
from itertools import combinations

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import ipywidgets as widgets
import rasterio
from rasterio import plot as rioplot
from rasterio.plot import show

# Machine learning metrics
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    recall_score, 
    precision_score, 
    precision_recall_curve
)

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def plot_training_distribution(training_gdf, by_area = False):

    """Plot distribution of training, validation and test sets for each LCZ class.
    
    Args:
        training_gdf (GeoDataFrame): GeoDataFrame containing training samples with 'name', 'set', and geometry columns
        by_area (bool): If True, calculate percentages by area instead of sample count
    
    Returns:
        plotly.graph_objects.Figure: Interactive stacked bar plot showing distribution of samples
    """
    # Calculate percentages
    
    
    if by_area:
        y_col = 'area_km2'
        training_gdf = training_gdf.to_crs(training_gdf.estimate_utm_crs())
        training_gdf[y_col]= round(training_gdf.area / 10**6, 2)

        counts = (training_gdf.groupby(['name', 'set'])
            [y_col].sum()
            .reset_index()
            .rename(columns={0:y_col}))
        total_counts = training_gdf.groupby('name')[y_col].sum()
        counts['percentage'] = counts.apply(lambda x: (x[y_col] / total_counts[x['name']]) * 100, axis=1)
        yaxis_title = 'Percentage of Total Area per LCZ Class (%)'
        text_col='percentage'
        texttemplate = '%{text:.0f}%'
    else:
        y_col = 'count'
        counts = (training_gdf.groupby(['name', 'set'])
            .size()
            .reset_index()
            .rename(columns={0:y_col}))
        
        total_counts = training_gdf.groupby('name').size()
        counts['percentage'] = counts.apply(lambda x: (x['count'] / total_counts[x['name']]) * 100, axis=1)
        yaxis_title = 'Percentage of Total Samples per LCZ Class (%)'
        text_col = 'count'
        texttemplate = '%{text:.0f}'
    class_order = training_gdf.sort_values('class')['name'].unique()

    # Create stacked bar plot
    fig = px.bar(counts, 
                x='name',
                y='percentage',
                color='set',
                text=text_col,
                color_discrete_map={'train': '#2ecc71',
                                'validation': '#3498db',
                                'test': '#e74c3c'},
                title='Distribution of Training/Validation/Test Sets by LCZ Class',
                category_orders={'name': class_order,  # Set the order of x-axis categories
                               'set': ['train', 'validation', 'test']}  # Set order of stacked bars
                )

    # Update layout
    fig.update_layout(
        barmode='stack',
        xaxis_title='LCZ Class',
        yaxis_title=yaxis_title,
        height=600,
        width=1000,
        showlegend=True,
        plot_bgcolor='white'
    )

    # Update axes
    fig.update_xaxes(
        tickangle=45,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        range=[0, 100],
        gridcolor='lightgrey'
    )
    # Add percentage labels inside bars
    fig.update_traces(
        texttemplate=texttemplate,
        textposition='inside',
        insidetextanchor='middle'
    )

    fig.show()

def plot_spectral_signature(band_stats, x_col, class_col, color_dict, title, xlabel, stat='median', out_file=None):
    """Plot spectral signature using a band statistics DataFrame 
    
    Args:
        band_stats (DataFrame): Pandas DataFrame of band statistics, retrieved from rasterstats.zonal_stats function
        x_col (str): Column for x-axis
        class_col (str): Column including class names
        color_dict (dict): Mapping between class name and color
        title (str): Plot title
        xlabel (str): Text label for x-axis
        stat (str): Name of statistic to plot, median is default, other options include mean, std, min and max
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Plot of spectral signature of all classess in the band_stats DataFrame
    """

    class_order = band_stats.drop_duplicates(class_col).sort_values(class_col)[class_col].to_list()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in class_order:
        class_stats=band_stats.set_index(class_col).loc[name]
        x = class_stats[x_col].to_list()
        y = class_stats[stat].to_list()
        ax.plot(x, y, label=name, color=color_dict[name]) # plot line for class

    ax.set_ylabel(f'{stat.title()} Surface Reflectance', weight = 'bold')
    ax.set_xlabel(xlabel, weight = 'bold')
    ax.set_title( title ,weight='bold')

    # Hide specific spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Legend formatting
    plt.legend(bbox_to_anchor =(0.5,-0.4), loc='lower center', ncol = 4, frameon=False)

    if out_file:
        plt.savefig(out_file)

    


def plot_pairwise_jm(df, class1,class2, dist_col, title, figsize, cbar=True,out_file=None):
    """Pairwise Jeffries-Matuista Distance Plot 
    
    Args:
        df (DataFrame): Jeffries-Matuista Distance between each class combination
        class1 (str): Column name of first classess
        class2 (str): Column name of second classes
        dist_col (str): Colum name including Jeffries-Matuista Distance values
        title (str): Plot title
        figsize (tuple): Desired figure Size
        cbar (boolean): Show color bar if True
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Plot of Jeffries-Matuista Distances between all classes
    """
    # Get Unique list of classes
    classes=df[class1].unique()
    arrays=[df.set_index(class1)[dist_col].loc[[cl]].values.T for cl in df[class1].unique()]

    max_len = max(len(arr) for arr in arrays)

    # Pad at the beginning (before index 0)
    result = [np.concatenate([np.zeros([max_len - len(arr)]),  arr])  for arr in arrays]

    matrix = np.vstack(result).T


    if out_file:
        plt.savefig(out_file)

    
    # Make pairwise distance plot
    plt.figure(figsize=figsize)
    plot_mask =  matrix == 0.0

    sns.heatmap(matrix, 
                annot=True, 
                fmt=".1f", 
                cmap='RdYlGn',
                xticklabels=classes, 
                yticklabels=classes, 
                linewidths=1, 
                linecolor='white', 
                # mask=plot_mask, 
                cbar=cbar
                
                )

    plt.title(title, fontweight='bold',fontdict=dict(size = 15), loc='left', x=0.0)

    if out_file:
        plt.savefig(out_file,bbox_inches='tight', dpi=300, pad_inches=0.2)



def plot_pixel_counts(pixel_count_df, count_col,class_col, color_col, title, as_percent=False, out_file=None):
    """Bar plot of pixel counts for each class
    
    Args:
        df (DataFrame): Pixel counts of each class in a Pandas DataFrame
        count_col (str): Name of column including pixel counts
        class_col (str): Name of column including class names
        color_col (str): Name of column including color values
        title (str): Plot title
        as_percent (boolean): Display values as percentage of total pixels if True
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Bar plot of pixel counts for each class
    """


    fig, ax = plt.subplots(figsize=(8, 6))
    
    pixel_count_df=pixel_count_df.sort_values(count_col,ascending=False) 
    y=pixel_count_df[count_col]  
    classes=pixel_count_df[class_col]
    colors=pixel_count_df[color_col]
    if as_percent:
        y = y / y.sum() * 100
        ylabel = 'Percentage of Pixels'
        plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0)) 
    else:
        
        ylabel = "Number of Pixels"

    

    ax.bar(x=classes,height=y, color=colors, linewidth=1, edgecolor='black')

    plt.xticks(rotation=90)
    plt.ylabel(ylabel, fontdict=dict(size=15))
    plt.title(title, fontdict=dict(size=15,weight='bold'))

    # Hide specific spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if out_file:
        plt.savefig(out_file)


def plot_confusion_matrix(y_true,y_pred, title,labels, figsize,cmap='Blues', as_percent=False, out_file=None):
    """Confusion matrix heat map using Seaborn 
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels 
        title (str): Plot title
        labels (list): Label names mapped to numeric label values
        figsize (tuple): Desired figure dimensions
        cmap (str): Desired color palette
        as_percent (boolean): Display values as percentage of total pixels if True
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Confusion matrix heat map using Seaborn 
    """

    cm=confusion_matrix(y_true,y_pred)
    mask = cm == 0.0

    
    plt.figure(figsize=figsize)
    fmt=".0f"

    if as_percent:
        cm=(cm / len(y_true)) * 100
        fmt=".1f"
    
    sns.heatmap(cm, 
                annot=True, 
                fmt=fmt, 
                cmap=cmap,
                cbar=False,
                xticklabels=labels, 
                yticklabels=labels, 
                # mask=mask,
                linewidths=1,
                linecolor='white'
             
                )
    plt.xlabel("True",  fontdict=dict(size = 15, weight ='bold'))
    plt.ylabel("Predicted",  fontdict=dict(size = 15, weight ='bold'))
    plt.title(title, fontdict=dict(size = 12, weight ='bold'), loc='center', pad=20, x= 0.4)
    plt.legend([],[], frameon=False)

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', )

    plt.show()





def plot_feature_importances(rf,features, title, out_file):
    """Plot Feature Importances of a Random Forest Classifier as a horizontal bar plot
    
    Args:
        rf (RandomForestClassifer): Trained RandomForestClassifer from sklearn
        features (list): Name of predictors use in the Random Forest Model
        title (str): Plot title
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Feature Importances of a Random Forest Classifier as a horizontal bar plot
    """

    rf_i = pd.Series(rf.feature_importances_, index=features).sort_values()

    # Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm', etc.)
    cmap = cm.get_cmap('RdYlGn')

    # Normalize the series to the range 0-1
    norm = colors.Normalize(vmin=rf_i.min(), vmax=rf_i.max())

    # Map each value to a color
    rgba_colors = [cmap(norm(val)) for val in rf_i]
    hex_colors = [colors.to_hex(c) for c in rgba_colors]

    fig, ax = plt.subplots()
    rf_i.plot.barh(ax=ax, color=hex_colors)
    ax.set_title(title)
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()

    if out_file:
        plt.savefig(out_file)

    # plt.show()
    return fig


def compare_feature_importances(cl1, cl2, names1, names2, title1, title2):
    """Compare feature importance plots between two classifiers side by side.
    
    Args:
        cl1 (RandomForestClassifier): First trained classifier
        cl2 (RandomForestClassifier): Second trained classifier
        names1 (list): Feature names for first classifier
        names2 (list): Feature names for second classifier
        title1 (str): Title for first plot
        title2 (str): Title for second plot
    
    Returns:
        matplotlib.figure.Figure: Side by side feature importance plots
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    rf_i = pd.Series(cl1.feature_importances_, index=names1).sort_values()

    # Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm', etc.)
    cmap = cm.get_cmap('RdYlGn')

    # Normalize the series to the range 0-1
    norm = colors.Normalize(vmin=rf_i.min(), vmax=rf_i.max())

    # Map each value to a color
    rgba_colors = [cmap(norm(val)) for val in rf_i]
    hex_colors = [colors.to_hex(c) for c in rgba_colors]

    rf_i.plot.barh(ax=ax1, color=hex_colors)
    ax1.set_title(title1)
    ax1.set_xlabel("Mean Decrease in Impurity")

    rf_i = pd.Series(cl2.feature_importances_, index=names2).sort_values()

    # Normalize the series to the range 0-1
    norm = colors.Normalize(vmin=rf_i.min(), vmax=rf_i.max())

    # Map each value to a color
    rgba_colors = [cmap(norm(val)) for val in rf_i]
    hex_colors = [colors.to_hex(c) for c in rgba_colors]

    rf_i.plot.barh(ax=ax2, color=hex_colors)
    ax2.set_title(title2)
    ax2.set_xlabel("Mean Decrease in Impurity")

    fig.tight_layout()

    fig.show()


def compare_confusion_matrices(matrix1, matrix2, plot_title, title1, title2, names, showscale=False, mask=False, cmap='Greens'):
    """Compare confusion matrices between two classifiers side by side.
    
    Args:
        matrix1 (ndarray): Confusion matrix for first classifier
        matrix2 (ndarray): Confusion matrix for second classifier
        plot_title (str): Overall plot title
        title1 (str): Title for first matrix
        title2 (str): Title for second matrix
        names (list): Class names for axis labels
        showscale (bool): Whether to show colorbar scale
        mask (bool): Whether to mask zero values
        cmap (str): Colormap to use for heatmap
    
    Returns:
        plotly.graph_objects.Figure: Interactive side by side confusion matrix plots
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(title1,title2),
                        horizontal_spacing=0.15)

    nanval = np.nan
    # Mask zero values with None for both matrices and text
    if mask:
        text1 = matrix1.astype(str)
        text2 = matrix2.astype(str)
        text1 = np.where(text1 == "0", "", text1)
        text2 = np.where(text2 == "0","", text2)


        matrix1 = np.where(matrix1 == 0.0, nanval, matrix1)
        matrix2 = np.where(matrix2 == 0.0, nanval, matrix2)
                
    else:
        text1 = matrix1
        text2 = matrix2

    # Add PRISMA heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix1,
            x=names,
            y=names,
            colorscale=cmap,
            zmin=np.min(matrix1),
            zmax=np.max(matrix1),
            text=text1,
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=showscale,
            colorbar=dict(x=0.45),
            xgap=2,  # Add gap between cells
            ygap=2,  # Add gap between cells
            hoverongaps=False  # Disable hover on gaps
        ),
        row=1, col=1
    )

    # Add Sentinel-2 heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix2,
            x=names,
            y=names,
            colorscale=cmap,
            zmin=np.min(matrix2),
            zmax=np.max(matrix2),
            text=text2,
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=showscale,
            colorbar=dict(x=1.0),
            xgap=2,  # Add gap between cells
            ygap=2,  # Add gap between cells
            hoverongaps=False  # Disable hover on gaps
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        showlegend=False,
        title_text=plot_title,
        template="simple_white",
        plot_bgcolor="white",
    )

    # Update axes
    for i in [1, 2]:
        fig.update_xaxes(tickangle=45, row=1, col=i)
        fig.update_yaxes(autorange="reversed", row=1, col=i)  # Reverse y-axis to match seaborn heatmap

    fig.show()


def plot_ovr_roc_comparison(cl1, cl2, X_test1, X_test2, y_test1, y_test2, cl1_name, cl2_name, class_dict, cols=4, figsize=(14,10)):
    """Compare One vs Rest ROC curves between two multi-class classifiers for each class.
        
    Args:
        cl1 (Classifier): First trained classifier
        cl2 (Classifier): Second trained classifier
        X_test1 (ndarray): Test features for first classifier
        X_test2 (ndarray): Test features for second classifier
        y_test1 (ndarray): Test labels for first classifier
        y_test2 (ndarray): Test labels for second classifier
        cl1_name (str): Name of first classifier
        cl2_name (str): Name of second classifier
        class_dict (dict): Mapping of class IDs to class names
        cols (int): Number of columns in subplot grid
        figsize (tuple): Figure size (width, height)
    
    Returns:
        matplotlib.figure.Figure: Grid of ROC curve comparisons for each class
    """
    classes = np.unique(y_test1)
    n_classes = len(classes)
    rows = int(np.ceil(n_classes / cols))

    ## Create Plot 
    plt.figure(figsize = figsize)
    # plt.title(f"ROC Curves of {n_classes} LCZs - One vs Rest", pad= 50, fontdict=dict(size = 20, weight='bold'))
    # Get Class Probabilities
    y_prob1 = cl1.predict_proba(X_test1)
    y_prob2 = cl2.predict_proba(X_test2)


    
    for i in range(len(classes)):
        
        c = classes[i]
        ## One vs Rest (OVR) Encoding for selected class
        binary1 = [1 if y == c else 0 for y in y_test1]
        binary2 = [1 if y == c else 0 for y in y_test2]

        # Retrieve predicted probabilites for selected class
        prob1 = y_prob1[:, i]
        prob2 = y_prob2[:, i]
    
        # Get Unique array of thresholds in from the class probabilities array
        thresholds1 = np.unique(prob1)
        thresholds2 = np.unique(prob2)

        # Compute TPR and FPR for selected class across all 
        tpr1, fpr1 = get_all_roc_coordinates(binary1,prob1, thresholds1)
        tpr2, fpr2 = get_all_roc_coordinates(binary2,prob2, thresholds2)

        # Compute Overall AUC OvR Score for the given Class
        score1 = roc_auc_score(binary1, prob1)
        label1 = f'{cl1_name} AUC = {round(score1,2)}'
        score2 = roc_auc_score(binary2, prob2)
        label2 = f'{cl2_name} AUC = {round(score2,2)}'

        plot = plt.subplot(rows,cols, i + 1)
        plot_roc_curve(tpr1, fpr1, ax = plot, color = 'blue', label=label1) # Plot ROC of first classifier
        plot_roc_curve(tpr2, fpr2, ax = plot, color = 'red', label=label2) # Plot ROC of second  classifier

        plot.set_title(f"LCZ {class_dict[c]} vs Rest (n= {n_classes - 1})", fontdict=dict(size = 12, weight = 'bold'))
        plot.legend(loc = 'lower right', fontsize = 'small')

    plt.tight_layout()


    
def calculate_tpr_fpr(y_real, y_pred):
    """Calculate True Positive Rate and False Positive Rate.
    
    Args:
        y_real (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
    
    Returns:
        tuple: (tpr, fpr) True Positive Rate and False Positive Rate
    """
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba, thresholds):
    """Calculate ROC curve coordinates at different thresholds.
    
    Args:
        y_real (array-like): Ground truth labels
        y_proba (array-like): Predicted probabilities
        thresholds (array-like): Classification thresholds to evaluate
    
    Returns:
        tuple: (tpr_list, fpr_list) Lists of TPR and FPR values at each threshold
    """
    tpr_list = [0]
    fpr_list = [0]
    for threshold in thresholds:
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list



def plot_roc_curve(tpr, fpr, ax = None, color ='blue', label=''):
    """Plot a single ROC curve.
    
    Args:
        tpr (array-like): True positive rates
        fpr (array-like): False positive rates
        ax (matplotlib.axes.Axes): Axes to plot on, creates new figure if None
        color (str): Line color
        label (str): Legend label
    
    Returns:
        matplotlib.axes.Axes: The axes containing the plot
    """
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
        
    sns.lineplot(x = fpr, y = tpr, ax = ax, color = color, label =label)
    sns.lineplot(x = [0, 1], y = [0, 1], linestyle='--', color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("FPR")
    plt.ylabel("TPR")



def plot_ovo_roc_comparison(cl1, cl2, X_test1, X_test2, y_test1,y_test2, cl1_name, cl2_name,class_dict, target_class, cols=10, figsize = (14, 10)):
    classes = list(np.unique(y_test1))
    
    ## Create Plot 
    plt.figure(figsize = figsize)

    # Get Class Probabilities
    y_prob1 = cl1.predict_proba(X_test1)
    y_prob2 = cl2.predict_proba(X_test2)

    # Create N x N class combinations    
    roc_classes = class_combinations(classes)
    roc_classes = [c for c in roc_classes if c[0] == target_class]
    n_classes = len(roc_classes)
    rows = int(np.ceil(n_classes / cols))

    for i in range(len(roc_classes)):
        
        c = roc_classes[i]
        c1, c2 = c
        mask1 = (y_test1 == c1) | (y_test1 == c2)
        mask2 = (y_test2 == c1) | (y_test2 == c2)
        class_index = list(classes).index(c1)

        ## One vs Rest (OVR) Encoding for selected class
        binary1 = [1 if y == c1 else 0 for y in y_test1[mask1]]
        binary2 = [1 if y == c1 else 0 for y in y_test2[mask2]]

        # Retrieve predicted probabilites for selected class
        prob1 = y_prob1[mask1, class_index]
        prob2 = y_prob2[mask2, class_index]
    
        # Get Unique array of thresholds in from the class probabilities array
        thresholds1 = np.unique(prob1)
        thresholds2 = np.unique(prob2)
        
        # Compute TPR and FPR for selected class across all 
        tpr1, fpr1 = get_all_roc_coordinates(binary1,prob1, thresholds1)
        tpr2, fpr2 = get_all_roc_coordinates(binary2,prob2, thresholds2)

        # Compute Overall AUC OvR Score for the given Class
        score1 = roc_auc_score(binary1, prob1)
        label1 = f'{cl1_name} AUC = {round(score1,2)}'
        score2 = roc_auc_score(binary2, prob2)
        label2 = f'{cl2_name} AUC = {round(score2,2)}'

        plot = plt.subplot(rows,cols, i + 1)
        plot_roc_curve(tpr1, fpr1, ax = plot, color = 'blue', label=label1) # Plot ROC of first classifier
        plot_roc_curve(tpr2, fpr2, ax = plot, color = 'red', label=label2) # Plot ROC of second  classifier

        plot.set_title(f"LCZ {class_dict[c1]} vs LCZ {class_dict[c2]}", fontdict=dict(size = 14, weight = 'bold'))
        plot.legend(loc = 'lower right', fontsize = 'small')

    plt.tight_layout()



def class_combinations(classes):
    classes_combinations = []
    class_list = list(classes)
    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):
            classes_combinations.append([class_list[i], class_list[j]])
            classes_combinations.append([class_list[j], class_list[i]])

    return classes_combinations


def plot_precision_recall_curve(precision, recall, ax, color, label):
        sns.lineplot(x = recall, y = precision, ax = ax, color = color, label = label)
        sns.lineplot(x = [0, 1], y = [0.5, 0.5], linestyle='--', color = 'black', ax = ax)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
  

def plot_precision_recall_comparison(cl1, cl2, X_test1, X_test2, y_test1,y_test2, cl1_name, cl2_name,name_dict, color_dict, figsize = (14, 10)):

    ## Create Plot 
    fig, ax = plt.subplots(1,2, figsize =figsize)
    classes = np.unique(y_test1)

    ## Get Class Predictions 
    y_pred1 = cl1.predict(X_test1)
    y_pred2 = cl2.predict(X_test2)
    # Get Class Probabilities
    y_prob1 = cl1.predict_proba(X_test1)
    y_prob2 = cl2.predict_proba(X_test2)

    ax1, ax2 = ax

    # Set titles for each subplot
    # Add main title for the entire figure
    fig.suptitle('Precision-Recall Curves', 
                fontsize=16, 
                fontweight='bold',
                y=0.95)  # Adjust y position of title

    ax1.set_title(f"{cl1_name} Classifer", fontdict = dict(weight = 'bold', size = 12))
    ax2.set_title(f"{cl2_name} Classifier", fontdict = dict(weight = 'bold', size = 12))

    # Set labels for each subplot
    ax1.set_xlabel("Recall",  fontdict = dict(weight = 'bold', size = 12))
    ax1.set_ylabel("Precision", fontdict = dict(weight = 'bold', size = 12))
    ax2.set_xlabel("Recall", fontdict = dict(weight = 'bold', size = 12))
    ax2.set_ylabel("Precision",  fontdict = dict(weight = 'bold', size = 12))



    for i in range(len(classes)):
        
        c = classes[i]
        ## One vs Rest (OVR) Encoding for selected class
        binary1 = [1 if y == c else 0 for y in y_test1]
        binary2 = [1 if y == c else 0 for y in y_test2]

        # Retrieve predicted probabilites for selected class
        prob1 = y_prob1[:, i]
        prob2 = y_prob2[:, i]
        # Compute TPR and FPR for selected class across all 
        
        # pre1, rec1 = get_pre_rec_coordinates(binary1,prob1, thresholds1)
        pre1, rec1, _ = precision_recall_curve(binary1,prob1)
        pre2, rec2, _ = precision_recall_curve(binary2,prob2)
        
        # plot Precision - Recall Surve of both classifiers
        plot_precision_recall_curve(pre1,rec1, ax = ax1, color=color_dict[c], label = name_dict[c])
        plot_precision_recall_curve(pre2,rec2, ax = ax2, color=color_dict[c], label = name_dict[c])

    # Compute Weighted Recall and Precision Score of Overall Model
    precision1 = precision_score(y_test1, y_pred1, average='weighted')
    precision2 = precision_score(y_test2, y_pred2, average='weighted')
    recall1 = recall_score(y_test1, y_pred1, average='weighted')
    recall2 = recall_score(y_test2, y_pred2, average='weighted')

    rx = 0.0
    ry = 0.1
    y_offset=0.06

    bbox=dict(boxstyle="round",
                   ec='black',
                   fc='white',
                   )
    fontdict   = dict(size = 12, weight='bold')
    ax1.text(x = rx, y = ry, s = f'Precision = {round(precision1, 2)}',  fontdict =fontdict , bbox = bbox)
    ax1.text(x = rx, y = ry - y_offset, s = f'Recall = {round(recall1, 2)}', fontdict = fontdict, bbox = bbox)

    ax2.text(x = rx, y = ry, s = f'Precision = {round(precision2, 2)}',  fontdict = fontdict, bbox = bbox)
    ax2.text(x = rx, y = ry - y_offset, s = f'Recall = {round(recall2, 2)}', fontdict =fontdict, bbox = bbox)
    
    
    # Remove individual legends
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    h1, l1 = ax1.get_legend_handles_labels()

    
    # Add single legend with unique items
    fig.legend(
        handles=h1,
        labels=l1,
        bbox_to_anchor=(0.5, 0.0),
        loc='upper center',
        ncol=5,
        fontsize='large',
        bbox_transform=fig.transFigure
    )

    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1,  # Space for legend at bottom
                    top=0.85)     # Space for main title at top
   