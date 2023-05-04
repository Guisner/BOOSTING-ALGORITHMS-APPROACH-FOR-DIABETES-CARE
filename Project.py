#Libraries used
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from warnings import filterwarnings
filterwarnings('ignore')

df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

#Detection of missing values (No missing values)
print(df.info())
print(df.isnull().sum())

#Detection of duplicate data (24206 duplicate data dropping from table)
print(df.duplicated().sum())
df.drop_duplicates(inplace = True)
print(df.duplicated().sum())
print(df.shape)


#Data visualization for diabetes (pie chart)
print(df.Diabetes_binary.value_counts())
labels=["non-Diabetic","Diabetic"]
plt.pie(df["Diabetes_binary"].value_counts() , labels =labels ,autopct='%.02f' );
plt.show()


#TESTING 
#Splitting data into training and test sets
X=df.drop("Diabetes_binary",axis=1)
y=df["Diabetes_binary"]

from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

#SMOTE
smote = SMOTE(random_state = 2)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train.ravel())

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
t0 = time()
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X_train, y_train)
gbm_time = time() - t0
gbm_acc = accuracy_score(y_test, gbm_model.predict(X_test))

#XGBoost
from xgboost import XGBClassifier
t0 = time()
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
XGBoost_time = time() - t0
XGBoost_acc = accuracy_score(y_test, xgb_model.predict(X_test))

#LightGBM
import lightgbm as lgb
t0 = time()
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_time = time() - t0
lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))

#Catboost
from catboost import CatBoostClassifier
t0 = time()
cat_model=CatBoostClassifier()
cat_model.fit(X_train, y_train)
cat_time = time() - t0
cat_acc = accuracy_score(y_test, cat_model.predict(X_test))

#Model Comparison numerically
print("Gradient Boosting: ", round(gbm_acc, 7), round(gbm_time, 2))
print("XGBoost: ", round(XGBoost_acc, 7), round(XGBoost_time, 2))
print("LightGBM: ", round(lgb_acc, 7), round(lgb_time, 2))
print("Catboost: ", round(cat_acc, 7), round(cat_time, 2))