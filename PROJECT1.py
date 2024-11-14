import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# STEP1 Data Acquisition

# Read data
amazon_sales = pd.read_csv('Amazon sales data.csv')

# Info of file
print(amazon_sales.info())

# Displaying few columns
print(amazon_sales.head())

#STEP2 EDA

# Clean the data
amazon_sales.dropna(inplace=True)

# Correct datatypes if necessary ( converting 'Order Date' to datetime)
amazon_sales['Order Date'] = pd.to_datetime(amazon_sales['Order Date'])

# Visualization using a pie chart of product categories 


plt.figure(figsize=(10, 6))
amazon_sales['Item Type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Product Categories")
plt.ylabel('')
plt.show()


#visualization using bar graph
plt.figure(figsize=(10, 6))
category_sales = amazon_sales.groupby('Item Type')['Total Profit'].sum().reset_index()
sns.barplot(x='Item Type', y='Total Profit', data=amazon_sales, hue='Item Type', palette='viridis', dodge=False)
plt.title("Total Sales per Category")
plt.xlabel("Item Type")
plt.ylabel("Total Profit")
plt.xticks(rotation=45)
plt.show()


#STEP3 DATA CLEANING AND PREPROCESSING
# Removing outliers
# Getting IQR for the 'Total Profit' column
Q1 = amazon_sales['Total Profit'].quantile(0.25)
Q3 = amazon_sales['Total Profit'].quantile(0.75)
IQR = Q3 - Q1

# Boundaries for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = amazon_sales[(amazon_sales['Total Profit'] >= lower_bound) & (amazon_sales['Total Profit'] <= upper_bound)]

print(f"The number of rows before removing the outliers: {amazon_sales.shape[0]}")
print(f"The number of rows after removing the outliers: {df_no_outliers.shape[0]}")


#STEP4 IN-DEPTH ANALYSIS AND VISUALIZATION

# Calculating total sales
total_sales = amazon_sales['Total Profit'].sum()
print(f"Total profit: {total_sales}")

# Calculating number of orders
num_orders = amazon_sales.shape[0]
print(f"The number of orders: {num_orders}")


#visualization using histogram
plt.figure(figsize=(10, 6))
sns.histplot(amazon_sales['Total Profit'], kde=True)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

#STEP5 Advanced Analysis and Hypothesis Testing

# Example of a hypothesis test: Chi-square test on categorical data, e.g., 'Category' and 'Region'
from scipy.stats import chi2_contingency

if 'Item Type' in amazon_sales.columns and 'Region' in amazon_sales.columns:
    contingency_table = pd.crosstab(amazon_sales['Item Type'], amazon_sales['Region'])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square Statistic: {chi2_stat}, P-Value: {p_value}")
else:
    print("Error: Columns 'Category' and/or 'Region' not found in the dataset.")


# Creating a correlation matrix 
numeric_cols = amazon_sales.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

# Display the correlation matrix
print(correlation_matrix)



#STEP6 TESTING MODULAR CODE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import unittest

# Function to load data
def load_data(df):
    try:
        df = pd.read_csv('Amazon sales data.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to preprocess data
def preprocess_data(df, features, target):
    """
    Preprocess the dataset by encoding categorical variables and handling missing values.
    
    Parameters:
    - df: DataFrame, the input dataset
    - features: list of str, the feature columns
    - target: str, the target column
    
    Returns:
    - X: DataFrame, the processed feature data
    - y: Series, the target data
    - label_encoders: dict, the label encoders used for encoding categorical variables
    """
    label_encoders = {}
    for col in features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    df.fillna(df.mean(), inplace=True)
    X = df[features]
    y = df[target]
    return X, y, label_encoders

# Function to split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Test class
class TestAmazonSalesAnalysis(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Product': ['A', 'B', 'A', 'C'],
            'Category': ['Electronics', 'Books', 'Electronics', 'Clothing'],
            'Region': ['North', 'South', 'East', 'West'],
            'Sales': [100, 200, 150, 300]
        })
        self.features = ['Product', 'Category', 'Region']
        self.target = 'Sales'

    def test_preprocess_data(self):
        X, y, label_encoders = preprocess_data(self.df, self.features, self.target)
        self.assertEqual(X.shape[1], len(self.features))
        self.assertEqual(len(y), len(self.df))

# Run tests
if __name__ == '__main__':
    unittest.main()



    
