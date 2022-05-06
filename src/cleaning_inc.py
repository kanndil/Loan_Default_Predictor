### Don't mind about this 
import warnings
#import seaborn as sns


warnings.filterwarnings('ignore')
###
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


table = pd.read_csv("Loan_Default.csv")

#preprocessing:

def label_encoding(label):
    uniques = table[label].unique()
    #the encoding is the index
    for i in range(len(uniques)):
        table.loc[table[label] == uniques[i],label]=i

table.drop('year', inplace=True, axis=1)
table.drop('Interest_rate_spread', inplace=True, axis=1)
table.drop('property_value', inplace=True, axis=1)
table.drop('dtir1', inplace=True, axis=1)
table.drop('ID', inplace=True, axis=1)
table.drop('Upfront_charges', inplace=True, axis=1)
table.drop('rate_of_interest', inplace=True, axis=1)
table.drop('LTV', inplace=True, axis=1)
income_mean = table['income'].mean()
table['income'] = table['income'].replace(np.nan, income_mean)
#label encoding
for i in range(len(table.iloc[3])):
   if(type(table.iloc[3][i])==str):
       label_encoding(table.columns[i])
# Normalization 
scaled_colmns={'loan_amount','term','income','Credit_Score'}
for column in scaled_colmns:
    table[column] = (table[column] - table[column].min()) / (table[column].max() - table[column].min())    

table_no_nulls = table.copy()
# Calulating the mean of both columns 
loan_limit_mean = table_no_nulls['loan_limit'].mean()
age_mean = table_no_nulls['age'].mean()

#replacing the values with the mean
table_no_nulls['age'] = table_no_nulls['age'].replace(np.nan, age_mean)
table_no_nulls['loan_limit'] = table_no_nulls['loan_limit'].replace(np.nan, loan_limit_mean)

#replacing NULL values with random numbers
import random

table_no_nulls['approv_in_adv'] = table_no_nulls['approv_in_adv'].apply(lambda x:x if pd.notnull(x) else random.randint(table_no_nulls['approv_in_adv'].min() , table_no_nulls['approv_in_adv'].max() ) )
table_no_nulls['submission_of_application'] = table_no_nulls['submission_of_application'].apply(lambda x:x if pd.notnull(x) else random.randint(table_no_nulls['submission_of_application'].min() , table_no_nulls['submission_of_application'].max() ) )
table_no_nulls['loan_purpose'] = table_no_nulls['loan_purpose'].apply(lambda x:x if pd.notnull(x) else random.randint(table_no_nulls['loan_purpose'].min() , table_no_nulls['loan_purpose'].max() ) )
table_no_nulls['Neg_ammortization'] = table_no_nulls['Neg_ammortization'].apply(lambda x:x if pd.notnull(x) else random.randint(table_no_nulls['Neg_ammortization'].min() , table_no_nulls['Neg_ammortization'].max() ) )
table_no_nulls['term'] = table_no_nulls['term'].apply(lambda x:x if pd.notnull(x) else random.randint(table_no_nulls['term'].min() , table_no_nulls['term'].max() ) )

# X_NoNull = table_no_nulls.loc[:,table_no_nulls.columns !="Status"] #all colomns except Status
# y_NoNull = table_no_nulls['Status'] #Status column

X = table_no_nulls.loc[:,table_no_nulls.columns !="Status"] #all colomns except Status
y = np.expand_dims(table_no_nulls['Status'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=30) #data is randomized during spliiting

# print ("x train",X_train)
# print ("y train",y_train)

# print("y_train original shape: ",y_train.shape)
