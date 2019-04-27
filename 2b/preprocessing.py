import pandas as pd
from sklearn import preprocessing
import numpy as np

def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df

def find_new_categories(original, altered):   
    newlist = list()
    for i in altered.keys():
        if not(i in original.keys()):
            newlist.append(i)
    return newlist

def normalize_category(data, column_name):
    data[column_name] = data[column_name].fillna(-0.5)
    x_array = np.array(data[column_name])
    normalized_X = preprocessing.normalize([x_array])
    normalized_X = pd.Series(normalized_X[0])

    data = pd.concat([data, normalized_X], axis=1)
    data.rename( columns={ 0 : column_name + '_new'}, inplace=True )
    print(data.columns)
    return data

def preprocess(data):

    # function to process the age of the passengers to categories
    # the min and max for the different categories
    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]


    # the labels for teh categories
    label_names = ["Missing", 'Infant', "Child",
               'Teenager', "Young Adult", 'Adult', 'Senior']

    # process the age to categories
    out = process_age(data, cut_points, label_names)

    # create dummies for different columns
    out = create_dummies(out, "Sex")
    out = create_dummies(out, "Pclass")
    out = create_dummies(out, "Age_categories")
    out = create_dummies(out, "Embarked")
    
    #out = create_dummies(out, "SibSp")

    #out = normalize_category(out, 'Fare')

    categories = find_new_categories(data, out)

    output = list()
    output.append(out)
    output.append(categories)

    return output

#0.8212290502793296
#0.811431165588469