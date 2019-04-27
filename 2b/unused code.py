import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# function to process the age of the passengers to categories
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

# the min and max for the different categories
cut_points = [-1,0, 5, 12, 18, 35, 60, 100]
# the labels for teh categories
label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

age_cat_pivot = train.pivot_table(index="Age_categories",values="Survived")
age_cat_pivot.plot.bar()
#plt.show()

# gets the number of duplicates for each value
train['Pclass'].value_counts()


#column_name = "Pclass"
#df = train
# create dummies to get around a numeric relationship in column
#dummies = pd.get_dummies(df[column_name],prefix=column_name)
#print(dummies.head())

# function to create dummies for one column and concat it with the table
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

# create dummies for different columns
train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")
train = create_dummies(train,"Sex")
test = create_dummies(test,"Sex")
train = create_dummies(train,"Age_categories")
test = create_dummies(test,"Age_categories")

# object for logic regression
lr = LogisticRegression()

# learn out model with columns
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
# function to learn and the prediction column
lr.fit(train[columns], train['Survived'])

lr.decision_function(train[columns])
lr.coef_

holdout = test # from now on we will refer to this
               # dataframe as the holdout data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.2,random_state=0)

    
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(test_y, predictions)
pd.DataFrame(conf_matrix, columns=['Survived', 'Died'], index=[['Survived', 'Died']])

from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(lr, all_X, all_y, cv=10)
np.mean(scores)


print('accuracy: ', accuracy)

print('mean :', np.mean(scores))

lr = LogisticRegression()
lr.fit(all_X, all_y)
holdout_predictions = lr.predict(holdout[columns])
print(holdout_predictions)

holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

print(submission)



holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv('titanic_submission.csv', index=False)

