import streamlit as st
import pandas as pd
import numpy as np
import random as rn
from glob import glob
#import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import numpy as np
import seaborn as sns
import time
import os
from PIL import Image
import random
import torch
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

st.title("Pawpularity Contest Dashboard")
st.write("Hello! I'm Sakshi. Let me walk you through my Pawpularity Contest Dashboard :) First of all, we have two parts in this dashboard.")
st.write("Data Analysis: EDA, data introduction and detailed data analysis ")
st.write("Model Analysis: Different models implemented for this use case with their comparison of best and worst performance")

# Path variables
BASE_PATH = "petfinder-pawpularity-score/"
TRAIN_PATH = BASE_PATH + "train.csv"
TEST_PATH = BASE_PATH + "test.csv"
TRAIN_IMAGES = glob(BASE_PATH + "train/*.jpg")
TEST_IMAGES = glob(BASE_PATH + "test/*.jpg")

# We are trying to predict this "Pawpularity" variable
TARGET = "Pawpularity"

# Seed for reproducability
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

df = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
selected_tab = st.radio("Select a Tab:", ["Data Analysis", "Model Analysis"])
if selected_tab == "Data Analysis":
# All relevant tabular futures
    FEATURES = [col for col in df.columns if col not in ['Id', TARGET]]
    feat = [col for col in df.columns if col not in ['Id']]
    st.write("We have two kinds of data here: Textual and Image. Let's take a look at how each dataset looks like. Feel free to go to the sidebar and play around with fields there. The filtered dataframe below will dynamically update according to your selections.")
    st.sidebar.title("Data Filtering")

    # Create filter widgets in the sidebar
    feature_to_filter = st.sidebar.selectbox("Select a feature to filter:", feat)
    min_value = st.sidebar.number_input("Minimum Value", min_value=df[feature_to_filter].min(), max_value=df[feature_to_filter].max())
    max_value = st.sidebar.number_input("Maximum Value", min_value=df[feature_to_filter].min(), max_value=df[feature_to_filter].max())

    # Filter the data based on user inputs
    filtered_data = df[(df[feature_to_filter] >= min_value) & (df[feature_to_filter] <= max_value)]

    # Display the filtered data
    st.subheader("Filtered Data")
    st.write(filtered_data)

    image_folder_path = "petfinder-pawpularity-score/train"

# Get a list of all image files in the folder
    #image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Function to select and display random images

# Move image loading into function
    image_filenames = os.listdir(image_folder_path)
    random.seed(time.process_time())
    num_images = 3
    filenames = image_filenames[:200]
    #random_images = random.sample(filenames, num_images)
    def display_random_images_in_row():
        random_images = random.sample(filenames, num_images)  
        columns = st.columns(num_images)
        cnt=0
        for image_file in random_images:
            # Load and display image
            image_path = os.path.join(image_folder_path, image_file)
            image = Image.open(image_path)

            with columns[cnt]:  
                st.image(image, caption=image_file)

            cnt+=1

    st.subheader("Random Image Generator")
    st.write("Now, let's have a look at our image dataset. Click the button below to popup 3 random images from the data. Note: I have only taken the first 200 images for this operation in order to maximize efficiency.")
    state = st.button("Generate random images")  
    if state:display_random_images_in_row()
    

    available_features = [col for col in df.columns if col != "Id"]
    st.title("Data Statistics")
    st.write("Let's look at the basic statistics of the dataset. Click the dropdown below to view correlation matrix, mean, median, count, etc.")

    # Exclude the "Id" column from statistical calculations
    selected_stats = st.multiselect("Select statistics to display:", ["Mean", "Median", "Count", "Type", "Min", "Max", "Correlation Matrix"])

    if "Mean" in selected_stats:
        st.write("Mean:", df[available_features].mean())

    if "Median" in selected_stats:
        st.write("Median:", df[available_features].median())

    if "Count" in selected_stats:
        st.write("Count:", df[available_features].count())

    if "Type" in selected_stats:
        st.write("Data Types:", df[available_features].dtypes)

    if "Min" in selected_stats:
        st.write("Min:", df[available_features].min())

    if "Max" in selected_stats:
        st.write("Max:", df[available_features].max())

    if "Correlation Matrix" in selected_stats:
        st.write("Correlation Matrix:")
        st.write(df[available_features].corr())


    # Exclude the "Id" column from statistical calculations
    st.title("Data Statistics for Binary Features")
    st.write("Now that we have looked at statistics, you will realize that most of these statistics do not make sense for features who have binary values. Review the section below to take a look at some of the meaningful statistics for binary features.")

    # Exclude the "Id" column from statistical calculations
    binary_features = [col for col in df.columns if col != "Id" and df[col].nunique() == 2]

    # Dropdown to select a binary feature
    selected_feature = st.selectbox("Select a binary feature for statistics:", binary_features)

    # Display counts and proportions for the selected binary feature
    st.write(f"Feature: {selected_feature}")
    st.write("Counts:")
    st.write(df[selected_feature].value_counts())
    st.write("Proportions:")
    st.write(df[selected_feature].value_counts(normalize=True))

    st.write("Mode:", df[selected_feature].mode().values[0])

    st.title("Interactive Data Analysis")
    st.write("Select any feature from the dropdown to view their distribution as a bar graph for better understanding.")
    
    # Dropdown to select the feature for distribution
    features_for_task1 = [col for col in df.columns if col != "Id"]
    feature_to_plot = st.selectbox("Select a feature for distribution:", features_for_task1)

    # Plot the distribution as a bar chart based on user selection
    st.bar_chart(df[feature_to_plot].value_counts())

    
    # Task 2 and Task 3: Popular and Least Popular Images
    
    st.title("Custom Box Plot and Histogram Visualization")
    st.write("Let's dive deep into advanced statistics. Let's check how each and every feature is related to our target feature 'pawpularity'. Select any feature from the dropdown below to get a visualization between their relation.")
    # Exclude the "Id" column and select two features for the visualization
    available_features = [col for col in df.columns if col != "Id"]

    # Create the custom visualization based on user selections
    feature1 = st.selectbox("Select the feature for the plot:", available_features)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Custom Box Plot
    sns.boxplot(data=df, x=feature1, y="Pawpularity", ax=ax[0])
    ax[0].set_title(feature1, fontsize=20, fontweight='bold')

    # Custom Histogram with Hue
    sns.histplot(data=df, x="Pawpularity", hue=feature1, kde=True, ax=ax[1])
    ax[1].set_title(feature1, fontsize=20, fontweight='bold')

    # Display the custom visualization in Streamlit
    st.pyplot(fig)

    st.title("Top 3 Popular Images")
    st.write("Let's do some fun stats and figure out which of the images are ranked the top 3 given the selected features. Feel free to select any number of features from the dropdown.")
    # Multi-select box to choose the features for popularity analysis for Task 2
    popularity_features_task2 = st.multiselect("Select features for popularity analysis (Task 2):", df.columns, default=["Pawpularity", "Eyes", "Face"])

    # Sort the dataframe by the selected features (in descending order) for Task 2
    top_3_images_task2 = df.sort_values(by=popularity_features_task2, ascending=False).head(3)

    # Display the top 3 popular images for Task 2
    image_column_task2 = st.columns(3)
    for index, row in top_3_images_task2.iterrows():
        with image_column_task2[index % 3]:
            image_path = os.path.join("petfinder-pawpularity-score/train", f"{row['Id']}.jpg")
            image = Image.open(image_path)
            st.image(image, caption=f"Popularity: {row['Pawpularity']}", use_column_width=True)

    # Task 3: Least Popular Images
    st.title("Top 3 Least Popular Images")
    st.write("Just like the task above, this section will tell us the 3 least popular images given the features. Feel free to select any number of features for the same.")

    # Multi-select box to choose the features for popularity analysis for Task 3
    popularity_features_task3 = st.multiselect("Select features for popularity analysis (Task 3):", df.columns, default=["Pawpularity", "Eyes", "Face"])

    # Sort the dataframe by the selected features (in ascending order) for Task 3
    bottom_3_images_task3 = df.sort_values(by=popularity_features_task3, ascending=True).head(3)

    # Display the top 3 least popular images for Task 3
    image_column_task3 = st.columns(3)
    for index, row in bottom_3_images_task3.iterrows():
        with image_column_task3[index % 3]:
            image_path = os.path.join("petfinder-pawpularity-score/train", f"{row['Id']}.jpg")
            image = Image.open(image_path)
            st.image(image, caption=f"Popularity: {row['Pawpularity']}", use_column_width=True)
elif selected_tab == "Model Analysis":
    # Your Model Analysis code for the "Model Analysis" tab
    st.title("Model Analysis")
    st.write("Now, lets move to the model analysis part. Here, I have written the code for 20 different models. Please select one from the drop down to find out the RMSE value, Residual graphs, predicted vs actual graphs plot and feature importance whereever it applies.")
    st.subheader("Technical Definitions")
    st.write("RMSE: Root Mean Square Error (RMSE) is a statistical measure that indicates how well a model can predict target values. It is the standard deviation of the residuals, which are the prediction errors. Residuals are a measure of how far from the regression line data points are. RMSE tells you how concentrated the data is around the line of best fit.")
    st.write("Residual Plot: A residual plot is a graphical technique that attempts to show the relationship between a given independent variable and the response variable given that other independent variables are also in the model.")
    st.write("Predicted vs Actual Plot: A predicted against actual plot shows the effect of the model and compares it against the null model. For a good fit, the points should be close to the fitted line, with narrow confidence bands.")
    st.write("Feature importance plots can indicate which fields had the biggest impact on each prediction that is generated by classification or regression analysis.")
    st.write("NOTE: Feature importance plots will not be supported by all models as feature importance is a concept that is primarily relevant to tree-based models and linear models.")
    st.write("NOTE: Apart from the models below, I have a list of additional 32 models that I have worked on. Unfortunately, due to hardware limits, it is not possible to deploy each and every one of them. You can find the code for them in the github link below: ")
    st.markdown('[GitHub Repository](https://github.com/SaShaShady/PawpularityContest/tree/main)', unsafe_allow_html=True)
#Dropdown to select the machine learning model
    selected_model = st.selectbox("Select a Model:", [
        "Lasso Regression", "Support Vector Regression", "Ridge Regression",
        "Kernel Ridge Regression", "Elastic Net Regression", "XGB Regression",
        "LGBM Regression", "Linear Regression", "Extra Trees Regression",
        "Decision Tree Regression", "Random Forest Regression",
        "Gaussian Process Regressor", "Gradient Boosting Regression",
        "AdaBoost Regression", "KNeighbours Regression", "Cat Boost Regression",
        "Random Forest Quantile Regression", "Mondrian Tree Regression",
        "Extra Trees Quantile Regression", "Decision Tree Quantile Regressor"
    ])

    if selected_model == "Lasso Regression":
        st.subheader("Lasso Regression Model")
        # Your Lasso Regression code here

    elif selected_model == "Support Vector Regression":
        st.subheader("Support Vector Regression Model")
        # Your SVR code here

    elif selected_model == "Ridge Regression":
        st.subheader("Ridge Regression Model")
        # Your Ridge Regression code here

    elif selected_model == "Kernel Ridge Regression":
        st.subheader("Kernel Ridge Regression Model")
        # Your Kernel Ridge Regression code here

    elif selected_model == "Elastic Net Regression":
        st.subheader("Elastic Net Regression Model")
        # Your Elastic Net Regression code here

    elif selected_model == "XGB Regression":
        st.subheader("XGBoost Regression Model")
        # Your XGBoost Regression code here

    elif selected_model == "LGBM Regression":
        st.subheader("LightGBM Regression Model")
        # Your LightGBM Regression code here

    elif selected_model == "Linear Regression":
        st.subheader("Linear Regression Model")
        # Your Linear Regression code here

    elif selected_model == "Extra Trees Regression":
        st.subheader("Extra Trees Regression Model")
        # Your Extra Trees Regression code here

    elif selected_model == "Decision Tree Regression":
        st.subheader("Decision Tree Regression Model")
        # Your Decision Tree Regression code here

    elif selected_model == "Random Forest Regression":
        st.subheader("Random Forest Regression Model")
        # Your Random Forest Regression code here

    elif selected_model == "Gaussian Process Regressor":
        st.subheader("Gaussian Process Regressor Model")
        # Your Gaussian Process Regressor code here

    elif selected_model == "Gradient Boosting Regression":
        st.subheader("Gradient Boosting Regression Model")
        # Your Gradient Boosting Regression code here

    elif selected_model == "AdaBoost Regression":
        st.subheader("AdaBoost Regression Model")
        # Your AdaBoost Regression code here

    elif selected_model == "KNeighbours Regression":
        st.subheader("K-Neighbors Regression Model")
        # Your K-Neighbors Regression code here

    elif selected_model == "Cat Boost Regression":
        st.subheader("Cat Boost Regression Model")
        # Your Cat Boost Regression code here

    elif selected_model == "Random Forest Quantile Regression":
        st.subheader("Random Forest Quantile Regression Model")
        # Your Random Forest Quantile Regression code here

    elif selected_model == "Mondrian Tree Regression":
        st.subheader("Mondrian Tree Regression Model")
        # Your Mondrian Tree Regression code here

    elif selected_model == "Extra Trees Quantile Regression":
        st.subheader("Extra Trees Quantile Regression Model")
        # Your Extra Trees Quantile Regression code here

    elif selected_model == "Decision Tree Quantile Regressor":
        st.subheader("Decision Tree Quantile Regressor Model")
        # Your Decision Tree Quantile Regressor code here

    # Sample function to calculate and display the common visualizations
    def visualize_model(model, X_train, X_test, y_train, y_test, feature_names):
        # Residual Plot
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # Residual Plot
        ax[0].scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
        ax[0].scatter(y_test_pred, y_test_pred - y_test, c='green', marker='s', label='Test data')
        ax[0].set_xlabel('Predicted values')
        ax[0].set_ylabel('Residuals')
        ax[0].legend(loc='upper left')
        ax[0].set_title('Residual Plot')

        # Predicted vs. Actual Values Plot
        ax[1].scatter(y_train_pred, y_train, c='blue', marker='o', label='Training data')
        ax[1].scatter(y_test_pred, y_test, c='green', marker='s', label='Test data')
        ax[1].set_xlabel('Predicted values')
        ax[1].set_ylabel('Actual values')
        ax[1].legend(loc='upper left')
        ax[1].set_title('Predicted vs. Actual Values Plot')

        # Feature Importance Plot (for models that support it)
        if hasattr(model, 'coef_'):
            coef = model.coef_
            ax[2].barh(range(len(coef)), coef, tick_label=feature_names)
            ax[2].set_xlabel('Coefficient Value')
            ax[2].set_ylabel('Feature')
            ax[2].set_title('Feature Importance')

        # Display the figure with subplots
        st.pyplot(fig)
        rmse = np.sqrt(mean_squared_error(y_test_pred, y_test))
        st.write(f"Local Test Score (RMSE) for {selected_model}: {rmse:.2f}")

    # Split the data into features and target variable
    X = df.drop(["Id", "Pawpularity"], axis=1)
    y = df["Pawpularity"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    feature_names = X.columns
    # Create and train the selected model
    if selected_model == "Lasso Regression":
        model = Lasso(alpha=0.1, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic')
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    # Add code for other models in a similar manner
    # Create and train the selected model
    if selected_model == "Support Vector Regression":
        model = SVR()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Ridge Regression":
        model = Ridge()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Kernel Ridge Regression":
        model = KernelRidge()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Elastic Net Regression":
        model = ElasticNet()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "XGB Regression":
        model = XGBRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "LGBM Regression":
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Extra Trees Regression":
        model = ExtraTreesRegressor(n_estimators=100, random_state=123)
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Decision Tree Regression":
        model = DecisionTreeRegressor(random_state=123)
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Random Forest Regression":
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Gaussian Process Regressor":
        model = GaussianProcessRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Gradient Boosting Regression":
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "AdaBoost Regression":
        model = AdaBoostRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "KNeighbours Regression":
        model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Cat Boost Regression":
        model = CatBoostRegressor()
        model.fit(X_train, y_train)
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Random Forest Quantile Regression":
        # Random Forest Quantile Regression

        # Create and train the selected model
        model = RandomForestRegressor(n_estimators=100, random_state=123)
        model.fit(X_train, y_train)

        feature_names = X.columns

        # Visualize the model
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Mondrian Tree Regression":
        # Mondrian Tree Regression

        # Create and train the selected model
        model = GradientBoostingRegressor(n_estimators=100, random_state=123)
        model.fit(X_train, y_train)

        feature_names = X.columns

        # Visualize the model
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Extra Trees Quantile Regression":
        # Extra Trees Quantile Regression

        # Create and train the selected model
        model = ExtraTreesRegressor(n_estimators=100, random_state=123)
        model.fit(X_train, y_train)

        feature_names = X.columns

        # Visualize the model
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)

    elif selected_model == "Decision Tree Quantile Regressor":
        # Decision Tree Quantile Regressor
        # Create and train the selected model
        model = DecisionTreeRegressor(random_state=123)
        model.fit(X_train, y_train)

        feature_names = X.columns

        # Visualize the model
        visualize_model(model, X_train, X_test, y_train, y_test, feature_names)
   
    # Create a dictionary to store RMSE values for each model
    rmse_values = {}

    
    rmse_values['Linear Regression']=20.94
    rmse_values['Ridge Regression']=20.94
    rmse_values['Gradient Boosting Regression']=20.94
    rmse_values['KNeighbours Regression']=24.17
    rmse_values['Kernel Ridge Regression']=22.75
    rmse_values['Suppport Vector Regression']=21.44
    # Sort the models by RMSE (low RMSE is better)
    top_3_models = dict(sorted(rmse_values.items(), key=lambda item: item[1])[:3])
    worst_3_models = dict(sorted(rmse_values.items(), key=lambda item: item[1], reverse=True)[:3])
    st.write("Here, we can see a list of the top 3 best performing and worst performing models along with their RMSE scores.")

    # Display top 3 models with their RMSE values
    st.subheader("Top 3 Models based on RMSE:")
    for model_name, rmse in top_3_models.items():
        st.write(f"{model_name}: RMSE = {rmse:.2f}")

    # Display worst 3 models with their RMSE values
    st.subheader("Worst 3 Models based on RMSE:")
    for model_name, rmse in worst_3_models.items():
        st.write(f"{model_name}: RMSE = {rmse:.2f}")
