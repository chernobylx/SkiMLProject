# Analysis of Ski Resort Lift Ticket Prices

## Project Description

This project aims to predict the price of a single-day ski resort lift ticket using geographical features and resort-specific attributes. Ski resorts are a cornerstone of winter tourism, and the price of lift tickets can vary significantly. By employing various predictive modeling techniques, this analysis provides valuable insights for both consumers and resort operators.

The study leverages a dataset from Kaggle to build and evaluate several regression models, including:
* Linear Regression (with feature selection and regularization)
* K-Nearest Neighbors (KNN)
* Decision Trees
* Random Forest
* AdaBoost

The performance of these models is compared to identify the most effective approach for forecasting lift ticket prices, offering a practical tool for market analysis and strategic decision-making. The final model, a **Linear Regression with 29 features selected via forward selection**, achieved the highest predictive accuracy with a **Test R² of 0.789**.

---

## Dataset

The data is sourced from the "Ski Resorts" dataset on Kaggle, provided by Ulrik Thyge Pedersen.

* **Source**: [Kaggle Ski Resorts Dataset](https://www.kaggle.com/datasets/ulrikthygepedersen/ski-resorts/data)
* **Content**: The dataset contains information on 499 ski resorts worldwide, with 24 features describing their location, size, lift infrastructure, and amenities. The target variable for this analysis is the `Price` in Euros for a single-day lift ticket.

---

## Exploratory Data Analysis (EDA) and Data Cleaning

The initial phase involved a thorough exploration and cleaning of the dataset to prepare it for modeling.

* **Initial State**: The dataset began with 499 records and 24 features.
* **Data Cleaning**:
    * Removed 9 resorts where the `Price` was listed as 0, as these were identified as missing entries.
    * Investigated resorts with 0 values for `Beginner slopes`, `Intermediate slopes`, or `Difficult slopes`. Notably, Aspen Mountain, a major resort, correctly has 0 registered beginner slopes.
    * The `Longest run` feature, which had 205 zero values, was determined to represent missing data rather than actual runs of 0 km. Due to the high proportion of missing entries, this feature was dropped.
    * One resort with 0 `Total lifts` and 0 `Lift capacity` was removed.
* **Feature Engineering**:
    * Two new features were created to better capture resort characteristics:
        1.  `Elevation`: The vertical drop of the resort, calculated as `Highest point` - `Lowest point`.
        2.  `Average_lift_capacity`: The average capacity per lift, calculated as `Lift capacity` / `Total lifts`.
* **Data Transformation**:
    * Categorical features (`Child friendly`, `Snowparks`, `Nightskiing`, `Summer skiing`) were converted from "Yes"/"No" strings to binary (1/0) integers.
    * Numerical features showed non-normal distributions and were normalized using `PowerTransformer`.
    * Geographical features (`Country`, `Continent`) were one-hot encoded to create dummy variables.

---

## Modeling and Results

A variety of regression models were trained and evaluated on the cleaned and preprocessed data. The data was split into a training set (80%) and a test set (20%).

The performance of each model was measured by its R-squared (R²) score on the test set.

| Model                                        | Test R² Score | Notes                                                                   |
| -------------------------------------------- | ------------- | ----------------------------------------------------------------------- |
| **Linear Regression (Forward Selection)** | **0.789** | **Best performing model with 29 features.** |
| Linear Regression (Ridge, α=0.92)            | 0.768         | Performed better than the full model and Lasso.                         |
| Random Forest Regressor                      | 0.765         | Best tree-based model. (max_leaf_nodes=190, n_estimators=260)           |
| Linear Regression (Full Model)               | 0.762         | Baseline linear model with all features.                                |
| Linear Regression (Lasso, α=0.002)           | 0.757         | Performed feature selection by shrinking some coefficients to zero.     |
| K-Nearest Neighbors (KNN)                    | 0.724         | Optimal performance with k=5 neighbors.                                 |
| AdaBoost Regressor                           | 0.675         | Performed worse than Random Forest. (learning_rate=2.69, n_estimators=650)|
| Decision Tree Regressor                      | 0.666         | Poorest performance among the tested models. (max_depth=6, max_leaf_nodes=28) |

### Conclusion

The analysis demonstrates that a **Linear Regression model using forward selection** provides the most accurate predictions for ski resort lift ticket prices on this dataset. The model's residuals showed a slight tendency to overpredict cheaper resorts and underpredict more expensive ones, indicating potential areas for future improvement.

---

## Usage and Installation

To replicate this analysis, please follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```

2.  **Install dependencies**:
    Ensure you have Python 3 and the following libraries installed:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn graphviz
    ```

3.  **Download the Dataset**:
    * Download the `resorts.csv` file from the [Kaggle Ski Resorts Dataset](https://www.kaggle.com/datasets/ulrikthygepedersen/ski-resorts/data).
    * Create a `data` directory in the project's root folder and place the `resorts.csv` file inside it.

4.  **Run the Notebook**:
    * Launch Jupyter Notebook or JupyterLab and open the `ski.ipynb` file to view and run the complete analysis.
