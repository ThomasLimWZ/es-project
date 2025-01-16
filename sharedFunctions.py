import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib_venn import venn2, venn3
import seaborn as sns
from IPython.display import display

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn . over_sampling import SMOTE

# Plot for dataframe
def display_df_details(df):
    # Gather the required information
    summary_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })

    summary_df['Unique Values List'] = [df[col].dropna().unique() for col in df.columns]

    # Apply the styling directly using a lambda function
    styled_df = (summary_df.style
        .apply(lambda s: ['color: red' if v > 0 else '' for v in s], subset=['Null Count'])
        # Chain the next styling for 'Dtype' column where the Dtype is 'object'
        .apply(lambda s: ['color: green' if str(v) == 'object' else '' for v in s], subset=['Dtype'])
        .set_properties(**{'text-align': 'left'})
    )
    
    styled_df = styled_df.set_properties(**{'text-align': 'left'})

    # Display the styled DataFrame
    display(styled_df)

# Print unique values for every object column
def print_unique_values_for_object_columns(df):
    columns = df.select_dtypes(include=['object']).columns
    # Display unique values of each categorical column
    for column in columns:
        values = df[column].unique()
        print(f"{column} : {values}")

def encode_objectdtypes_columns(df, label_encode_columns):
    for column in df.columns:
        # converting all to lowercase when it is string data type
        unique_values_lower = set(x.lower() if isinstance(x, str) else x for x in df[column].unique())
        # Apply mapping for the specific column
        if unique_values_lower == {"yes", "no"}:
            df[column] = df[column].str.lower().map({"yes": 1, "no": 0})

    # Label encoding columns using LabelEncoder()
    label_encoder = LabelEncoder()
    for column in label_encode_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df

# ******************************************* Data Exploration Funtions ****************************************************
def plot_categorical_columns(df, columns):
    categorical_columns = columns

    # Check if specified columns exist in the DataFrame
    missing_columns = set(categorical_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
        
    # Check if the list of categorical columns is empty
    if not categorical_columns:
        raise ValueError("List of categorical columns is empty.")
        
    # Calculate the number of rows and columns dynamically
    num_cols = 2
    num_rows = math.ceil(len(categorical_columns) / num_cols)

    # Set up subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 2 * num_rows))
    fig.subplots_adjust(hspace=1, bottom=0.1)  # Adjust bottom space

    # Loop through each categorical column and create bar plots
    for i, column in enumerate(categorical_columns):
        row, col = divmod(i, num_cols)
        sns.countplot(x=column, data=df, ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {column}')

    # If the last subplot is empty, remove it
    if len(categorical_columns) % num_cols != 0:
        empty_subplots = num_cols - (len(categorical_columns) % num_cols)
        for i in range(1, empty_subplots + 1):
            fig.delaxes(axes[-1, -i])

    # Display the plots
    plt.show()

# Plot to check is data is balanced
def class_distribution(df, target_column, color_palette=["#512b58", "#fe346e"]):
    if color_palette is None:
        color_palette = ['#4b4b4c', '#512b58']

    class_count = pd.Series(df[target_column]).value_counts()

    plt.figure(figsize=(8, 6))
    ax_after = class_count.plot(kind='bar', color=color_palette)
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')

    ax_after.set_xticks([])
    ax_after.set_xticklabels([])

    ax_after.set_yticks([])
    ax_after.set_yticklabels([])

    # Annotate each bar with its count
    for i, count in enumerate(class_count):
        ax_after.text(i, count + 0.1, str(count), ha='center', va='bottom')

    # Create a custom legend
    legend_labels = ['Non-Stroke', 'Stroke']
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_palette]
    ax_after.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(1.1, 1), title='Legend')

    plt.tight_layout()
    plt.show()

# Plot Categorical data with stroke
def plot_categorical_stroke(df, column_name):
    stroke_cat = df[df['stroke'] == 1][column_name].value_counts()
    healthy_cat = df[df['stroke'] == 0][column_name].value_counts()

    unique_values = df[column_name].unique()

    stroke_values = [int(round(stroke_cat[val] / df[column_name].value_counts()[val] * 100, 0)) for val in unique_values]
    healthy_values = [int(round(healthy_cat[val] / df[column_name].value_counts()[val] * 100, 0)) for val in unique_values]

    total_values = df[column_name].value_counts().values
    total_percentage = [int(round(val / df.shape[0] * 100, 0)) for val in total_values]

    data = {
        column_name: unique_values,
        'Stroke': stroke_values,
        'No Stroke': healthy_values
    }

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    bar_positions1 = range(len(df_plot))
    bar_positions2 = [pos + bar_width for pos in bar_positions1]

    ax.bar(bar_positions1, df_plot['Stroke'], width=bar_width, label='Stroke', color='#fe346e')
    ax.bar(bar_positions2, df_plot['No Stroke'], width=bar_width, label='No Stroke', color='#512b58')

    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions1])
    ax.set_xticklabels(df_plot[column_name])
    ax.set_ylabel('Percentage')
    ax.set_title(f'{column_name.capitalize()} Risk for Stroke', fontdict={'font': 'Serif', 'size': 15, 'weight': 'bold', 'color': 'black'})
    ax.legend()

    for i, value in enumerate(df_plot['Stroke']):
        ax.text(bar_positions1[i], value + 1, f'{value}%', ha='center', va='bottom', fontweight='bold', color='#fe346e')

    for i, value in enumerate(df_plot['No Stroke']):
        ax.text(bar_positions2[i], value + 1, f'{value}%', ha='center', va='bottom', fontweight='bold', color='#512b58')

    plt.show()

def plot_stacking_bars(df, x, y, title='Title leh Haloooo', color_palette=['#512b58', '#fe346e']):

    # Pivot the data to get counts for each combination A and B
    course_depression_counts = df.pivot_table(index=x, columns=y, aggfunc='size', fill_value=0)

    # Set color palette
    sns.set_palette(color_palette)

    # Plot the stacked bar chart
    ax_after = course_depression_counts.plot(kind='bar', stacked=True, figsize=(15, 15))
    plt.title(title, fontdict={'font': 'Serif', 'size': 20, 'weight': 'bold', 'color': 'black'})
    
    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Create a custom legend
    legend_labels = ['Non-Stroke', 'Stroke']
    legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_palette]
    ax_after.legend(legend_handles, legend_labels, loc='upper right', title='Legend')

    plt.show()

def plot_stroke_percentage(df, title, subtitle):
    x = pd.DataFrame(df.groupby(['stroke'])['stroke'].count())

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=70)
    ax.barh([1], x.stroke[1], height=0.7, color='#fe346e')
    plt.text(-1150, -0.08, 'Healthy', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'style': 'normal',
                                       'color': '#512b58'})
    plt.text(5000, -0.08, '95%', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'color': '#512b58'})
    ax.barh([0], x.stroke[0], height=0.7, color='#512b58')
    plt.text(-1000, 1, 'Stroke', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'style': 'normal',
                                 'color': '#fe346e'})
    plt.text(300, 1, '5%', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'color': '#fe346e'})

    fig.patch.set_facecolor('#f6f5f5')
    ax.set_facecolor('#f6f5f5')

    plt.text(-1150, 1.77, title, {'font': 'Serif', 'size': '25', 'weight': 'bold','color': 'black'})
    plt.text(4650, 1.65, 'Stroke ', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'weight': 'bold', 'style': 'normal',
                                     'color': '#fe346e'})
    plt.text(5650, 1.65, '|', {'color': 'black', 'size': '16', 'weight': 'bold'})
    plt.text(5750, 1.65, 'Healthy', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'style': 'normal',
                                     'weight': 'bold', 'color': '#512b58'})
    plt.text(-1150, 1.5, subtitle,
             {'font': 'Serif', 'size': '12.5', 'color': 'black'})

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

def plot_stroke_percentage_df2(df2, title, subtitle):
    # Calculate stroke counts
    x = df2['stroke'].value_counts()
    percentage_stroke = ((x[1] / (x[0] + x[1])) * 100).round(2)
    percentage_of_healthy = 100 - percentage_stroke
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=70)

    # Bar for Healthy
    ax.barh([1], x[0], height=0.7, color='#512b58')
    plt.text(-5150, 1, 'Stroke', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'style': 'normal', 'color': '#fe346e'})
    plt.text(21000, 1, f'{percentage_stroke}%', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'color': '#fe346e'})

    # Bar for Stroke
    ax.barh([0], x[1], height=0.7, color='#fe346e')
    plt.text(-5150, 0, 'Healthy', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'style': 'normal', 'color': '#512b58'})
    plt.text(21000, 0, f'{percentage_of_healthy}%', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'color': '#512b58'})

    # Styling
    fig.patch.set_facecolor('#f6f5f5')
    ax.set_facecolor('#f6f5f5')

    plt.text(-5150, 1.77, title, {'font': 'Serif', 'size': '25', 'weight': 'bold', 'color': 'black'})
    plt.text(25650, 1.65, 'Stroke ', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'weight': 'bold', 'style': 'normal', 'color': '#fe346e'})
    plt.text(25000, 1.65, '|', {'color': 'black', 'size': '16', 'weight': 'bold'})
    plt.text(20000, 1.65, 'Healthy', {'font': 'Serif', 'weight': 'bold', 'size': '16', 'style': 'normal', 'weight': 'bold', 'color': '#512b58'})
    plt.text(-5150, 1.5, subtitle, {'font': 'Serif', 'size': '12.5', 'color': 'black'})

    # Adjust axes visibility
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

def plot_comparison_stroke(df, x, hue, title, xlabel=None, ylabel=None):
    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = hue
        
    color_palette=['#512b58', '#fe346e']

    sns.set_palette(color_palette)
    
    # Set the background color of the figure
    fig, ax = plt.subplots(figsize=(15, 15), facecolor='#f5f5f5')
    
    sns.countplot(x=df[x], hue=df[hue], data=df, ax=ax)
    
    ax.set_title(title, fontdict={'font': 'Serif', 'size': 20, 'weight': 'bold', 'color': 'black'})
    ax.set_xlabel(xlabel, fontdict={'font': 'Serif', 'size': 15, 'weight': 'bold', 'color': 'black'})
    ax.set_ylabel(ylabel, fontdict={'font': 'Serif', 'size': 15, 'weight': 'bold', 'color': 'black'})
    
    ax.tick_params(axis='x', rotation=0)
    
    # Create a custom legend
    legend_labels = ['Non-Stroke', 'Stroke']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in color_palette]
    ax.legend(legend_handles, legend_labels, loc='upper right', title='Legend')

    plt.show()
    return

def plot_venn(subsets, set_labels, set_colors, title="Venn Diagram", alpha=0.9):
    num_subsets = len(subsets)

    if num_subsets == 2:
        venn_function = venn2
    elif num_subsets == 3:
        venn_function = venn3
    else:
        raise ValueError("The function supports only 2 or 3 subsets.")

    venn_function(subsets=subsets, set_labels=set_labels, set_colors=set_colors, alpha=alpha)
    plt.title(title, fontsize=16)
    plt.show()

# Correlation Heatmap
def plot_correlation_heatmap(df):
    colors = ['#f6f5f5','#512b58','#fe346e']
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    corr = sns.clustermap(df.corr(), annot=True, fmt='0.2f',
                    cbar=False, cbar_pos=(0, 0, 0, 0), linewidth=0.5,
                    cmap=colormap, dendrogram_ratio=0.1,
                    facecolor='#f6f5f5', figsize=(8, 8),
                    annot_kws={'font': 'serif', 'size': 10, 'color': 'black'})

    plt.gcf().set_facecolor('#f6f5f5')
    label_args = {'font': 'serif', 'font': 18, 'weight': 'bold'}
    plt.setp(corr.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=10, fontfamily='Serif', fontweight='bold', alpha=0.8)  # For y-axis
    plt.setp(corr.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=10, fontfamily='Serif', fontweight='bold', alpha=0.8)  # For x-axis
    corr.fig.text(0, 1.065, 'Visualization of Clustering of Each Feature with Other', {'font': 'serif', 'size': 16, 'weight': 'bold'})
    corr.fig.text(0, 1.015, 'Lines on the top and left of the cluster map are called \ndendrograms, which indicate the dependency of features.', {'font': 'serif', 'size': 12}, alpha=0.8)
    plt.show()

# ******************************************* Data Preprocessing Funtions ****************************************************
# Split the data, return X_train, X_test, X_valid, y_train, y_test, y_valid
def split_data(df, target_column, apply_standard_scaler=False):
    # Split the data into train, test, and validation sets
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    if apply_standard_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

# Handle imbalanced data, return X_res_df as resampled data
def SMOTE_resample(df, target_column):
    # Separate the features and target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=101)
    X_res, y_res = sm.fit_resample(X, y)

    # Combine the resampled data into a DataFrame
    X_res_df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_column)], axis=1)
    
    # Shuffle the data
    X_res_df = X_res_df.sample(frac=1, random_state=101).reset_index(drop=True)
    return X_res_df

# ******************************************* Error Analysis Funtions ****************************************************
# Function to plot the ROC curve for classifiers
def plot_all_roc_curves(predictions, dataset_key, prediction_type):
    plt.figure(figsize=(10, 8))

    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Unpack only the needed elements from each tuple in the dictionary
    for i, model_info in enumerate(predictions[dataset_key][prediction_type]):
        model_name = model_info[0]
        y_pred = model_info[2]
        y_true = model_info[3]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})', color=color_palette[i])

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {prediction_type}')
    plt.legend(loc="lower right")
    plt.show()

# Plot deep learning model history
def plot_history(history):
    """
    Plots the training and validation loss from a Keras model's history.
    
    Parameters:
        history: A Keras History object returned from the fit method.
        
    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

# ******************************************* Model Comparison Funtions ****************************************************   
# Function to plot the comparison for a specific dataset and mental health condition
def plot_comparison(predictions, dataset, condition):
    models = [entry[0] for entry in predictions[dataset][condition]]
    accuracies = [entry[4] for entry in predictions[dataset][condition]]

    # Define different colors for each bar
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=colors, alpha=0.6)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_title(f'Model Comparison for {condition} in Dataset {dataset}', fontsize=16)
    plt.ylim([0, 1.2])
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)

    # Display the accuracy values on top of the bars
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=12)

    # Display model names outside the bars
    for bar, model in zip(bars, models):
        height = bar.get_height()

    # Add a legend for the colors
    ax.legend(bars, models, title='Models', title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()

# Function to plot confusion matrices for a specific dataset and mental health condition
def plot_confusion_matrices(predictions, dataset, condition, vmin=-0.5, vmax=1):
    models = [entry[0] for entry in predictions[dataset][condition]]
    num_models = len(models)
    num_rows = (num_models + 2) // 3  # Ensure there are enough rows to fit all models

    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(12, num_rows * 4))
    fig.patch.set_facecolor('#f6f5f5')

    # Setting of axes; visibility of axes and spines turned off
    for ax_list in axes:
        for ax in ax_list:
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.set_facecolor('#f6f5f5')

    for i in range(num_rows * 3):
        if i < len(predictions[dataset][condition]):
            y_pred = predictions[dataset][condition][i][2]
            y_true = predictions[dataset][condition][i][3]
            alg = models[i]
            cf = confusion_matrix(y_true, y_pred)
            # print('model ', i, ' : ', models[i], 'cf : ', cf)
            acc = predictions[dataset][condition][i][4]

            # Annotations for confusion matrix
            labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
            counts = ["{0:0.0f}".format(value) for value in cf.flatten()]
            percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
            label = (np.array([f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(labels, counts, percentages)])).reshape(2, 2)

            # Heatmap for confusion matrix
            sns.heatmap(data=cf, vmin=vmin, vmax=vmax, cmap=['grey'], linewidth=2, linecolor='#f6f5f5',
                        ax=axes.flatten()[i], annot=label, fmt='', cbar=False,
                        annot_kws={'font': 'serif', 'size': 10, 'color': 'white', 'weight': 'bold'}, alpha=0.8)

            # Subtitle
            axes.flatten()[i].text(0, -0, 'Confusion Matrix - {}'.format(alg), {'font': 'serif', 'size': 12, 'color': 'black', 'weight': 'bold'})

            # Accuracy plotting
            if acc > 0.9:
                # print(123)
                axes.flatten()[i].scatter(1, 1, s=3500, c='#fe346e')
                axes.flatten()[i].text(0.85, 1.1, 'Acc:\n{:.1f}%'.format(acc * 100),
                                    {'font': 'serif', 'size': 12, 'color': 'black', 'weight': 'bold'})
            else:
                axes.flatten()[i].scatter(1, 1, s=3500, c='white')
                axes.flatten()[i].text(0.85, 1.1, 'Acc:\n{:.1f}%'.format(acc * 100),
                                    {'font': 'serif', 'size': 12, 'color': 'black', 'weight': 'bold'})
        else:
            ax.axis('off')
    plt.show()

def plot_feature_importance(predictions, dataset, condition, df):
    # Get the list of models
    selected_classifiers = ['RF', 'Decision Tree', 'LightGBM', 'XGBoost', 'CatBoost']
    
    # Get the list of models
    models = [entry[0] for entry in predictions[dataset][condition]]

    # Filter out only the selected classifiers
    selected_models = [model for model in models if model in selected_classifiers]

    if dataset == 2:
        num_rows = len(selected_models)
        num_cols = 1
        figsize = (12, num_rows * 5)  # Adjust the figure size accordingly
    else:
        num_rows = len(selected_models) // 2 + len(selected_models) % 2
        num_cols = 2
        figsize = (12, num_rows * 5)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, model in enumerate(selected_models):
        ax = axes[i]
        
        # Find the index of the model in the original list
        index = models.index(model)

        # Get the classifier from the predictions
        classifier = predictions[dataset][condition][index][1]

        # Check if the classifier has feature_importances_ attribute
        if hasattr(classifier, 'feature_importances_'):
            # Extract feature importances
            feature_importances = classifier.feature_importances_

            # Get the feature names from DataFrame
            feature_names = df.columns

            # Plot feature importance
            ax.bar(range(len(feature_importances)), feature_importances, alpha=0.5, color='orange')
            ax.set_title(f'Feature Importance for {model} - {condition}')
            ax.set_ylabel('Feature Importance', color='orange')
            ax.tick_params('y', colors='orange')
            ax.set_xticks(range(len(feature_importances)))
            ax.set_xticklabels(feature_names, rotation=90, ha='right')  # Use feature names for x-axis labels
        else:
            ax.axis('off')  # Turn off axes for classifiers without feature_importances_

    # If there's an odd number of models, leave the last subplot empty
    if len(selected_models) % 2 != 0 and dataset != 2:
        axes[-1].axis('off')

    plt.tight_layout()
    plt.show()

# Display formatted metrics
def display_metrics(metrics):
    # Find the maximum length of 'Best Parameter' across all models
    max_param_length = max(len(str(details['Best Parameter'])) for details in metrics.values())
    
    # Define the minimum width for the 'Best Parameter' column
    min_param_width = 20
    # Ensure the column is at least the minimum width
    best_param_width = max(max_param_length, min_param_width)
    
    # Print the header
    print(f"{'Model':<15} {'Best Parameter':<{best_param_width}} {'Accuracy'}")
    print('-' * (15 + best_param_width + 10))
    
    # Print each row
    for model, details in metrics.items():
        best_param = str(details['Best Parameter'])
        accuracy = details['Accuracy']
        print(f"{model:<15} {best_param:<{best_param_width}} {accuracy:.2f}")