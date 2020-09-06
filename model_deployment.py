# Script to deploy model locally

# import Flask and jsonify
from flask import Flask, jsonify, request
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import numpy
import pickle
import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion


# Goal is to build an API that will tell us loan probabilites when it receives the information. Will use Flask
app = Flask(__name__)
api = Api(app)

# Import custom built made classes so pickle object can communicate with them 

def catFeat(data):
    cat_feats = data.dtypes[data.dtypes == 'object'].index.tolist()
    #num_feats = data.dtypes[~data.dtypes.index.isin(cat_feats)].index.tolist()
    return data[cat_feats]

# Create above function into a FunctionTransformer
keep_cat = FunctionTransformer(catFeat)

# Impute missing values with our own transformer
class CategoricalFillerTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, df, **transform_params):
        gen = df.Gender.value_counts(normalize=True)
        gen_missing = df['Gender'].isnull()
        df.loc[gen_missing,'Gender'] = np.random.choice(gen.index, size=len(df[gen_missing]),p=gen.values)
        
        marr = df.Married.value_counts(normalize=True)
        marr_missing = df['Married'].isnull()
        df.loc[marr_missing,'Married'] = np.random.choice(marr.index, size=len(df[marr_missing]),p=marr.values)
        
        dep = df.Dependents.value_counts(normalize=True)
        dep_missing = df['Dependents'].isnull()
        df.loc[dep_missing,'Dependents'] = np.random.choice(dep.index, size=len(df[dep_missing]),p=dep.values)
        
        self_emp = df.Self_Employed.value_counts(normalize=True)
        self_emp_missing = df['Self_Employed'].isnull()
        df.loc[self_emp_missing,'Self_Employed'] = np.random.choice(self_emp.index, size=len(df[self_emp_missing]),p=self_emp.values)
        
        return df        
    
    def fit(self, data, y=None, **fit_params):
        return self
    

# One Hot Encode to grab dummy variables 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')

# don't forget ToDenseTransformer after one hot encoder
class ToDenseTransformer():

    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self
    
to_dense = ToDenseTransformer()


def numFeat(data):
    cat_feats = data.dtypes[data.dtypes == 'object'].index.tolist()
    num_feats = data.dtypes[~data.dtypes.index.isin(cat_feats)].index.tolist()
    return data[num_feats]

# Create above function into a FunctionTransformer
keep_num = FunctionTransformer(numFeat)


# Create Feature Transformer on select columns (only numerical in our case)
class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, data, **transform_params):
        data['LogLoanAmount'] = np.log(data['LoanAmount'])
        data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
        data['LogApplicantIncome'] = np.log(data['ApplicantIncome'])
        data['LogTotalIncome'] = np.log(data['ApplicantIncome'] + data['CoapplicantIncome'])
        return data        
    
    
    def fit(self, data, y=None, **fit_params):
        return self
    
# Impute missing values after feature engineering with the median
from sklearn.impute import SimpleImputer
null_replace_num = SimpleImputer(strategy="median")

# Load Model
with open("/Users/Shrey/LHL_Notes/Week_7/Day_4/mini-project-IV/pickle_model.pkl", 'rb') as file:
    model = pickle.load(file)
    
    
# Now, we need to create an endpoint where we can communicate with our ML model. This time, we are going to use POST request.
class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        test = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        convert_dict = {'ApplicantIncome': int, 
                'CoapplicantIncome': float,
                'LoanAmount': float,
                'Loan_Amount_Term' : float,
                'Credit_History' : float
               } 
        test = test.astype(convert_dict) 
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict_proba(test)
        status = 'First value probability of not getting a loan and second value is probability of getting a loan'
        # we cannot send numpt array as a result
        return res.tolist() , status
    
    
# assign endpoint
api.add_resource(Scoring, '/scoring')


# The last thing to do is to create an application run when the file api.py is run directly (not imported as a module from another script).

if __name__ == '__main__':
    app.run(debug=True)

