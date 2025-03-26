# Import Libraries

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import warnings
from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Projects/Feature/First One/Womens Clothing E-Commerce Reviews.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

# Data Understanding

def check_data(dataframe):
    print("######## Shape ########")
    print(dataframe.shape)
    print("######## NA ########")
    print(dataframe.isnull().sum())
    print("######## Types ########")
    print(dataframe.dtypes)
    print("######## Unique ########")
    print(dataframe.nunique())

check_data(df)

df.head()

def get_col_names(dataframe, cat_th = 10, car_th = 20):
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","object","bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int64","float64"]]

    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category","object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = get_col_names(df)

df.groupby("Clothing ID").agg({"Recommended IND" : ["mean","count"]}).head(10)
df.groupby("Clothing ID").agg({"Recommended IND" : ["mean","count"]}).shape

df["Clothing ID"].value_counts().head(10)

df.groupby(["Clothing ID","Rating"]).agg({"Recommended IND" : ["mean","count"]}).head(10)
df.groupby(["Clothing ID","Rating"]).agg({"Recommended IND" : ["mean","count"]}).shape

cat_cols = [col for col in cat_cols if col != "Recommended IND"]

def cat_analyze(dataframe, cat_cols):
    print(pd.DataFrame({cat_cols: dataframe[cat_cols].value_counts(),
                       "Ratio" : dataframe[cat_cols].value_counts() * 100 / len(df)}))

for col in cat_cols:
    cat_analyze(df, col)

def num_analyze(dataframe, num_cols):
    quantiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)
    print("#"*15)

num_cols = [col for col in num_cols if col != "Clothing ID"]

for col in num_cols:
    num_analyze(df,col)

def target_analiz_cat(dataframe, cat_cols, target):
    print(pd.DataFrame({cat_cols: dataframe.groupby(cat_cols)[target].mean()}))

cat_cols= [col for col in cat_cols if col != "Recommended IND"]

for col in cat_cols:
    target_analiz_cat(df, col, "Recommended IND")

def target_analiz_num(dataframe, num_cols, target):
    print(dataframe.groupby(target)[num_cols].mean())

for col in num_cols:
    target_analiz_num(df, col, "Recommended IND")


# Let's see some graphics

sns.countplot(x = "Rating", palette="Paired", data=df)
plt.title("Rating Distribution", size=15)
plt.xlabel("Ratings")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="Department Name", palette="Paired", data=df)
plt.title("Department Name", size=15)
plt.show()

fig, axes = plt.subplots(1, len(num_cols), figsize=(20, 8))
for i, col in enumerate(num_cols):
    axes[i].hist(df[col])
    axes[i].set_title(col)
plt.show()

palette = sns.color_palette("Set2")

plt.figure(figsize=(12, 8))
for i, column in enumerate(num_cols, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=df[column], color=palette[i-1], width=0.5)
    plt.axhline(y=df[column].median(), color="green", linestyle="--", linewidth=2)
    plt.title(f"Boxplot of {column}", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

## Outlier

def outlier_thresholds(dataframe, col_name, quantile1 = 0.01, quantile3 = 0.99):
    q1 = dataframe[col_name].quantile(quantile1)
    q3 = dataframe[col_name].quantile(quantile3)
    iqr = q3 - q1
    up = quantile3 + 1.5 * iqr
    low = quantile1 - 1.5 * iqr
    return low, up

def check_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outliers(df, col))

outlier_thresholds(df,"Positive Feedback Count")
outlier_thresholds(df,"Age")

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outliers(df, col))

cat_cols, num_cols, cat_but_car = get_col_names(df)

num_cols=[col for col in num_cols if  col !="Clothing ID"]

df[num_cols].describe().T

df.columns= [col.upper() for col in df.columns]
df.columns

# New_Features

df["NEW_AGE"]=pd.cut(df["AGE"], bins=[18,50,max(df["AGE"])] ,labels=["mature","senior"])
df["NEW_RATING"]= pd.cut(df["RATING"],bins=[float('-inf'), 2, 3, float('inf')],labels=["bad","good","perfect"])

def new_features(dataframe,selected_col,selected_col_2,new_name):
    for i in dataframe[selected_col].value_counts().index:
        for j in dataframe[selected_col_2].value_counts().index:
            df.loc[(dataframe[selected_col]==i) & (dataframe[selected_col_2]==j),new_name]= i+"_"+j

new_features(df,"NEW_AGE","DIVISION NAME","NEW_AGE_DIVISION")
new_features(df,"NEW_AGE","DEPARTMENT NAME","NEW_AGE_DEPARTMENT")
new_features(df,"NEW_AGE","CLASS NAME","NEW_AGE_CLASS")
new_features(df,"NEW_RATING","DIVISION NAME","NEW_RATING_DIVISION")
new_features(df,"NEW_RATING","DEPARTMENT NAME","NEW_RATING_DEPARTMENT")
new_features(df,"NEW_RATING","CLASS NAME","NEW_RATING_CLASS")
new_features(df,"NEW_AGE_DEPARTMENT","NEW_RATING","NEW_AGE_DEPARTMENT_RATING")
new_features(df,"NEW_AGE_CLASS","NEW_RATING","NEW_AGE_CLASS_RATING")

def new_features_2(dataframe,selected_col,selected_col_2,new_name):
    for i in dataframe[selected_col].value_counts().index:
        for j in dataframe[selected_col_2].value_counts().index:
            df.loc[(dataframe[selected_col]==i) & (dataframe[selected_col_2]==j),new_name]= str(i) +"_"+j

new_features_2(df,"CLOTHING ID","NEW_RATING","NEW_CLOTHES_RATING")
new_features(df,"NEW_CLOTHES_RATING","NEW_AGE","NEW_CLOTHES_RATING_AGE")
df["RATING_POSITIVE"]=df["RATING"]* df["POSITIVE FEEDBACK COUNT"]
df["ID_POSITIVE"]=str(df["CLOTHING ID"])+"_"+str(df["POSITIVE FEEDBACK COUNT"])
df["CLOTHES_ID_RATING"]=str(df["CLOTHING ID"])+"_"+ str(df["RATING"])

df.head()

df.drop("NEW_CLOTHES_RATING", axis=1, inplace=True)
df.drop("NEW_AGE_CLASS", axis=1, inplace=True)
df.drop(["NEW_AGE_DIVISION","NEW_CLOTHES_RATING_AGE"], axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = get_col_names(df)

cat_cols=[col for col in cat_cols if col!= "RECOMMENDED IND"]

num_cols

cat_but_car

cat_cols = cat_cols + cat_but_car + ["CLOTHING ID"]
df.head()

# One Hot

def one_hot_encoder(dataframe, cat_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = cat_cols, drop_first = drop_first)
    return dataframe

df = one_hot_encoder(df,cat_cols)

num_cols = [col for col in num_cols if col != "CLOTHING ID"]

# Standardization

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df[num_cols].head()

# Model

y = df["RECOMMENDED IND"]
X = df.drop("RECOMMENDED IND", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

## Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
rf_accuracy= accuracy_score(y_pred, y_test)
rf_accuracy

## KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

knn_model= KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred =knn_model.predict(X_test)
knn= accuracy_score(y_pred, y_test)
knn

##

def plot_importance(model, features, num=40, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)