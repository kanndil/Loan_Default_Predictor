import streamlit as st
import pandas as pd

from loan_default_predictor import NN



st.write ('''

# Welcome to The Loan Delfault Predictor

''')

st.sidebar.header("Bank client input data")
#

def get_new_sample():
    loan_limit= st.sidebar.selectbox('loan_limit', ('cf', 'ncf'), index=1)
    Gender= st.sidebar.selectbox('Gender', ('Male', 'Female'), index=1)
    approv_in_adv= st.sidebar.selectbox('approv_in_adv', ('pre', 'nopre'), index=1)
    loan_type= st.sidebar.selectbox('loan_type', ('type1', 'type2', 'type3'), index=1)
    loan_purpose= st.sidebar.selectbox('loan_purpose', ('p1', 'p2', 'p3','p4'), index=1)
    Credit_Worthiness= st.sidebar.selectbox('Credit_Worthiness', ('l1', 'l2'), index=1)
    open_credit= st.sidebar.selectbox('open_credit', ('opc', 'nopc'), index=1)
    business_or_commercial= st.sidebar.selectbox('business_or_commercial', ('b/c', 'nob/c'), index=1)
    loan_amount= st.sidebar.slider('loan_amount' ,16500,3576500,1000000)
    term=st.sidebar.slider('term' ,96,360,200)
    Neg_ammortization= st.sidebar.selectbox('Neg_ammortization', ('neg_amm', 'not_neg'), index=1)
    interest_only= st.sidebar.selectbox('interest_only', ('int_only', 'not_int'), index=1)
    lump_sum_payment= st.sidebar.selectbox('lump_sum_payment', ('lpsm', 'not_lpsm'), index=1)
    construction_type= st.sidebar.selectbox('construction_type', ('mh', 'sb'), index=1)
    occupancy_type= st.sidebar.selectbox('occupancy_type', ('ir', 'pr', 'sr'), index=1)
    Secured_by=  st.sidebar.selectbox('Secured_by', ('home', 'land'), index=1)
    total_units= st.sidebar.selectbox('total_units', ('1U', '2U', '3U','4U'), index=1)
    income= st.sidebar.slider('income' ,0,500000,10000)
    credit_type= st.sidebar.selectbox('credit_type', ('CIB', 'CRIF', 'EQUI','EXP'), index=1)
    Credit_Score= st.sidebar.slider('Credit_Score' ,300,900,500)
    co_applicant_credit_type= st.sidebar.selectbox('co_applicant_credit_type', ('CIB','EXP'), index=1)
    age= st.sidebar.selectbox('age', ('<25','25-34', '35-44','45-54','55-64','65-74', '>74'), index=1)
    submission_of_application= st.sidebar.selectbox('submission_of_application', ('to_inst','not_inst'), index=1)
    Region= st.sidebar.selectbox('Region', ('central','North', 'North-East', 'south'), index=1)
    Security_Type= st.sidebar.selectbox('Region', ('direct','indirect'), index=1)




    loan_amount_c= (loan_amount-16500.0)/(3576500.0-16500.0)
    term_c= (term-96.0)/(360.0-96.0)
    income_c= (income)/(578580.0)
    Credit_Score_c= (Credit_Score-300.0)/(900.0-300.0)


#['cf' nan 'ncf']
#['Sex Not Available' 'Male' 'Joint' 'Female']
#['nopre' 'pre' nan]
#['type1' 'type2' 'type3']
#['p1' 'p4' 'p3' 'p2' nan]
#['l1' 'l2']
#['nopc' 'opc']


    if (loan_limit=='cf'):      loan_limit_c=0 
    elif (loan_limit=='ncf'):   loan_limit_c=2

    if (Gender=='Male'):        Gender_c=1
    elif (Gender=='Female'):    Gender_c=3

    if (approv_in_adv=='nopre'):       approv_in_adv_c=0
    elif (approv_in_adv=='pre'):       approv_in_adv_c=1

    if   (loan_type=='type1'):       loan_type_c=0
    elif (loan_type=='type2'):       loan_type_c=1
    elif (loan_type=='type3'):       loan_type_c=3

    if   (loan_purpose=='p1'):       loan_purpose_c=0
    elif (loan_purpose=='p2'):       loan_purpose_c=3
    elif (loan_purpose=='p3'):       loan_purpose_c=2
    elif (loan_purpose=='p4'):       loan_purpose_c=1  
    
    if   (Credit_Worthiness=='l1'):  Credit_Worthiness_c=0
    elif (Credit_Worthiness=='l2'):  Credit_Worthiness_c=1

    if   (open_credit=='opc'):   open_credit_c=1
    elif (open_credit=='nopc'):  open_credit_c=0

#['nob/c' 'b/c']
#['not_neg' 'neg_amm' nan]
#['not_int' 'int_only']
#['not_lpsm' 'lpsm']
#['sb' 'mh']
#['pr' 'sr' 'ir']
#['home' 'land']
#['1U' '2U' '3U' '4U']
#['EXP' 'EQUI' 'CRIF' 'CIB']
#['CIB' 'EXP']


    if   (business_or_commercial=='b/c'):    business_or_commercial_c=0
    elif (business_or_commercial=='nob/c'):  business_or_commercial_c=1

    if   (Neg_ammortization=='neg_amm'):    Neg_ammortization_c=1
    elif (Neg_ammortization=='not_neg'):    Neg_ammortization_c=0

    if   (interest_only=='int_only'):    interest_only_c=1
    elif (interest_only=='not_int'):     interest_only_c=0

    if   (lump_sum_payment=='lpsm'):            lump_sum_payment_c=1
    elif (lump_sum_payment=='not_lpsm'):        lump_sum_payment_c=0

    if   (construction_type=='mh'):            construction_type_c=1
    elif (construction_type=='sb'):            construction_type_c=0

    if   (occupancy_type=='ir'):            occupancy_type_c=2
    elif (occupancy_type=='pr'):            occupancy_type_c=0
    elif (occupancy_type=='sr'):            occupancy_type_c=1

    if   (total_units=='1U'):            total_units_c=0
    elif (total_units=='2U'):            total_units_c=1
    elif (total_units=='3U'):            total_units_c=2
    elif (total_units=='4U'):            total_units_c=3

    if   (Secured_by=='home'):            Secured_by_c=0
    elif (Secured_by=='land'):            Secured_by_c=1

    if   (credit_type=='CIB'):            credit_type_c=3
    elif (credit_type=='CRIF'):           credit_type_c=2
    elif (credit_type=='EQUI'):           credit_type_c=1
    elif (credit_type=='EXP'):            credit_type_c=0

    if   (co_applicant_credit_type=='CIB'):   co_applicant_credit_type_c=0
    elif (co_applicant_credit_type=='EXP'):   co_applicant_credit_type_c=1

#['25-34' '55-64' '35-44' '45-54' '65-74' '>74' '<25' nan]
#['to_inst' 'not_inst' nan]
#['south' 'North' 'central' 'North-East']
#['direct' 'Indriect']

    if   ( age =='<25'):             age_c=6
    elif ( age =='25-34'):           age_c=0
    elif ( age =='35-44'):           age_c=2
    elif ( age =='45-54'):           age_c=3
    elif ( age =='55-64'):           age_c=1
    elif ( age =='65-74'):           age_c=4
    elif ( age =='>74'):             age_c=5


    if   (submission_of_application=='to_inst'):    submission_of_application_c=0
    elif (submission_of_application=='not_inst'):   submission_of_application_c=1

    if   (Region=='central'):       Region_c=2
    elif (Region=='North'):         Region_c=1
    elif (Region=='North-East'):    Region_c=3
    elif (Region=='south'):         Region_c=0

    if   (Security_Type=='direct'):    Security_Type_c=0
    elif (Security_Type=='indirect'):   Security_Type_c=1


    data= {
                'loan_limit':loan_limit_c,
                'Gender':Gender_c,
                'approv_in_adv':approv_in_adv_c,
                'loan_type': loan_type_c,
                'loan_purpose': loan_purpose_c,
                'Credit_Worthiness': Credit_Worthiness_c,
                'open_credit': open_credit_c,
                'business_or_commercial': business_or_commercial_c,
                'loan_amount': loan_amount_c,
                'term': term_c,
                'Neg_ammortization': Neg_ammortization_c,
                'interest_only': interest_only_c,
                'lump_sum_payment': lump_sum_payment_c,
                'construction_type': construction_type_c,
                'occupancy_type': occupancy_type_c,
                'Secured_by': Secured_by_c,
                'total_units': total_units_c,
                'income': income_c,
                'credit_type': credit_type_c,
                'Credit_Score': Credit_Score_c,
                'co-applicant_credit_type': co_applicant_credit_type_c,
                'age': age_c,
                'submission_of_application': submission_of_application_c,
                'Region': Region_c,
                'Security_Type': Security_Type_c,
    }


    features= pd.DataFrame(data, index=[0])
    return features


@st.experimental_singleton
def cleaning_and_fitting():

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

    model_classifier = NN(lr=0.4)

    model_classifier.add_layer(25,8,activation="sigmoid",name="l1")
    model_classifier.add_layer(8,8,activation = "sigmoid",name="l2")
    model_classifier.add_layer(8,8,activation = "sigmoid",name="l3")
    model_classifier.add_layer(8,1,activation = "sigmoid",name="l4")


    ############
    model_classifier.fit(X_train.astype(float),y_train.astype(float),epochs=200)



    return model_classifier





#main //////////////////////////////////

new_sample  = get_new_sample()

st.write(new_sample)

model_classifier=cleaning_and_fitting()

predicted_label = model_classifier.predict(new_sample)



#prediction_output=''
#if (prediction==1):
#    prediction_output="User will default"
#else :
#    prediction_output="User will not default"


st.subheader('Prediction')

st.write(predicted_label)

