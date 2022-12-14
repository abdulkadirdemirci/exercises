import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import pydotplus
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, \
    mean_squared_error, mean_absolute_error, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, validation_curve, KFold, \
    StratifiedKFold, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export_text
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler



pd.set_option("display.expand_frame", True)
pd.set_option("display.max_columns", 500)

df_ = pd.read_csv("datasets/churn.csv")
df = df_.copy()

# 1. Exploring data
df.describe([.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]).T
df.shape
df.isnull().sum()
df.describe()
df.sample(10)
df.nunique()
df.info()
df["Account_Manager"].unique()

for col_name in df.iloc[:, 1:].columns:
    print(df[col_name].value_counts(), end="\n\n\n")

for col_name in df.iloc[:, 1:].columns:
    print(df.groupby("Churn").agg({col_name: ["mean", "count"]}), end="\n\n\n")

for col_name in df.iloc[:, 1:].columns:
    df[col_name].plot(kind="hist", bins=20)
    plt.title(f"{col_name} freqoency graph")
    plt.show(block=True)

for col_name in df.iloc[:, 1:].columns:
    sns.boxplot(data=df, x="Churn", y=col_name, palette="magma")
    plt.show()

"""
it is a inbalanced dataset in the scope of churn variable
750 - 150 StratifiedKFold or KFold should be applied

maybe rare encoding applied ? i'aint sure 'bout that
"""

########################################
# 2. Preprocessing & Feature Engineering
########################################


#######################
# ordinal new variables
#######################

# yaşı ortalamanın altnıda olup total_purchase'i ort üstü olanlar
df.loc[(df["Age"] < df.Age.mean()) & (df["Total_Purchase"] > df.Total_Purchase.mean()), "New_var1"] = 4  # genc_iyi
df.loc[(df["Age"] > df.Age.mean()) & (df["Total_Purchase"] > df.Total_Purchase.mean()), "New_var1"] = 3  # yaslı_iyi
df.loc[(df["Age"] < df.Age.mean()) & (df["Total_Purchase"] < df.Total_Purchase.mean()), "New_var1"] = 2  # genç_kötü
df.loc[(df["Age"] > df.Age.mean()) & (df["Total_Purchase"] < df.Total_Purchase.mean()), "New_var1"] = 1  # yaslı_kötü

# years ort. altı olup yüksek total_purchase olanlar
df.loc[(df["Years"] < df.Years.mean()) & (df["Total_Purchase"] > df.Total_Purchase.mean()), "New_var2"] = 4  # yeni_iyi
df.loc[(df["Years"] > df.Years.mean()) & (df["Total_Purchase"] > df.Total_Purchase.mean()), "New_var2"] = 3  # eski_iyi
df.loc[(df["Years"] < df.Years.mean()) & (df["Total_Purchase"] < df.Total_Purchase.mean()), "New_var2"] = 2  # yeni_kötü
df.loc[(df["Years"] > df.Years.mean()) & (df["Total_Purchase"] < df.Total_Purchase.mean()), "New_var2"] = 1  # eski_kötü

# numsites düşük olan total_purchase yüksek olan
df.loc[(df["Num_Sites"] < df.Num_Sites.mean()) & (
        df["Total_Purchase"] > df.Total_Purchase.mean()), "New_var3"] = 4  # az_site - iyi_yatırım
df.loc[(df["Num_Sites"] > df.Num_Sites.mean()) & (
        df["Total_Purchase"] > df.Total_Purchase.mean()), "New_var3"] = 3  # çok_site - iyi_yatırım
df.loc[(df["Num_Sites"] < df.Num_Sites.mean()) & (
        df["Total_Purchase"] < df.Total_Purchase.mean()), "New_var3"] = 2  # az_site - kötü_yatırım
df.loc[(df["Num_Sites"] > df.Num_Sites.mean()) & (
        df["Total_Purchase"] < df.Total_Purchase.mean()), "New_var3"] = 1  # çok_site - kötü_yatırım

######################
# nominal new variable
######################

# yası küçük olan year buyuk olan
df["New_var4"] = df["Years"] / df["Age"]

# total_purchase / (abs(ort-age)*years*num_sites)
df["New_var5"] = df["Total_Purchase"] / (abs(df["Age"].mean() - df["Age"]) * df["Years"] * df["Num_Sites"])

df.head()

"""
bir önceki adıma tekrar gidip outlier oluşmuş mu kontrol et.

! New_var5 da outliers var baskılayacagız.
"""


########################
# new_var5 baskılama
########################

# alt ve üst sınırları bul
def get_limit(df, col_name, q1, q3):
    q1 = df[col_name].quantile(q1)
    q3 = df[col_name].quantile(q3)

    iqr = q3 - q1

    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


low_limit, up_limit = get_limit(df, "New_var5", 0.05, 0.95)


# aykırıları sınırlarla değiştir
def replace_with_threshols(df, col_name, low, up):
    temporary_df = df.copy()

    temporary_df.loc[temporary_df[col_name] > up_limit, col_name] = up
    temporary_df.loc[temporary_df[col_name] < low_limit, col_name] = low

    return temporary_df


df = replace_with_threshols(df, "New_var5", low_limit, up_limit)

###########
# encoding & standartization
###########
"""
encoding işlemine uygun kategorik değişkneler bulunmuyor
sadece standartizasyon uygulayacagız. aykırılıkları baskıladık ama yine de 
aykırılıklara bagışıklığı olan robust scaler kullanıyorum
"""
df1 = df.iloc[:, 1:]

rscaler = RobustScaler()

df1_scaled = pd.DataFrame(rscaler.fit_transform(df1), columns=df1.columns)

#########################
# 3. Modeling using CART
#########################
"""
bagımlı değişken: Churn

bagımsız değişkenler:'Total_Purchase', 'Account_Manager', 'Years',
                     'Num_Sites', 'New_var1', 'New_var2', 'New_var3',
                     'New_var4', 'New_var5'
"""
X = df1_scaled.drop("Churn", axis=1)
y = df1_scaled["Churn"]

#############
# model setup
#############

# model hiperparametrelerine bak hangileri üzerinde
# değişiklik yapacagına karar ver
DecisionTreeClassifier()._get_param_names()

set_params = {"criterion": ["gini", "entropy", "log_loss"],
              "max_depth": range(3, 10),
              "min_sample_split": range(2, 10)}

# boş model nesnesi oluşturuldu
cart_model = DecisionTreeClassifier()

# inbalanced data set old. için stratify parametresini giriyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

y_train.value_counts()
# 0: 562
# 1: 113

y_test.value_counts()
# 0: 188
# 1: 37

# train 0/1 = 4.97       test 0/1 = 5.081

# modeli train ile eğitmek

cart_model.fit(X_train, y_train)

# model train tahmini
y_train_pred = cart_model.predict(X_train)
y_train_prob = cart_model.predict_proba(X_train)[:, 1:]

# Train hatasi
classification_report(y_train, y_train_pred)
confusion_matrix(y_train, y_train_pred)
accuracy_score(y_train, y_train_pred)
recall_score(y_train, y_train_pred)
precision_score(y_train, y_train_pred)
f1_score(y_train, y_train_pred)
roc_auc_score(y_train, y_train_prob)
y_train_prob = cart_model.predict_proba(X_train)[:, 1:]

# model test tahmini
y_test_pred = cart_model.predict(X_test)
y_test_prob = cart_model.predict_proba(X_test)[:, 1:]

# test hatasi
classification_report(y_test, y_test_pred)
confusion_matrix(y_test, y_test_pred)
accuracy_score(y_test, y_test_pred)
recall_score(y_test, y_test_pred)
precision_score(y_test, y_test_pred)
f1_score(y_test, y_test_pred)
roc_auc_score(y_test, y_test_prob)

"""
GENEL YORUM:

evet overfitted cart_model hayırlı olsun.
Accuracy test setinde yüksek çıktı bunun sebebi
veri setinin inbalanced olması. şimdi akıllarda
biz inbalanced old. biliyorduk bu yuzden train test split'de
stratify=y yaptık sunuc bu mu? sorusu gelebilir. 
fakat bizim ordaki amacımız modeli öğrenme aşamasında yanlı değil
bagmlı ve bağımsız degişkenlerin eşit oranda bulundugu bir train test veri sesti ile
beslemekti. budagılımları kontrol ettiğimiz aşamada da kendinz de fark etmişsinizdir.
inbalanced durumu y_test ve y_trainde oransal olarak korunuyor. biz neyi engelledik peki?
y_train'in tamamen churn=0 y_test'in de tamamen churn=1 değerlerinden oluşmasını engelledik.

ayrıca bu başarı metrikleri hala çok ham. 
henüz hiperparametre optimizasyonu yapılmadı ve model validasyonu yapılmadı

model validasyonu hiperparametre opt. den sonra yapılacak.
"""

####################################################
# 4. Hyperparameter Optimization with GridSearchCV
####################################################
stf = StratifiedKFold(5)

cart_params = {"criterion": ["gini", "entropy", "log_loss"],
               "max_depth": range(3, 10),
               "min_samples_split": range(2, 10)}

best_cart_model = GridSearchCV(cart_model,
                               cart_params,
                               cv=stf,
                               n_jobs=-1,
                               verbose=2)

best_cart_model.fit(X_train, y_train)

best_cart_model.best_params_
# criterion : GİNİ
# max_depth: 3
# min_samples_split: 2

################
# 5. Final Model
################

# gridsearchcv modeli çıktısına göre tekrar model oluşturuldu
best_cart_model_tuned = DecisionTreeClassifier(**best_cart_model.best_params_)
# parametre kontrolü
best_cart_model_tuned.get_params()

best_cart_model_tuned.fit(X_train, y_train)
y_pred = best_cart_model_tuned.predict(X_test)
y_prob = best_cart_model_tuned.predict_proba(X_test)[:, 1:]

# başarı metrikleri kontrolu(cross valide edilmemiş veri üzerinde)
print(classification_report(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)
accuracy_score(y_test, y_test_pred)
recall_score(y_test, y_test_pred)
precision_score(y_test, y_test_pred)
f1_score(y_test, y_test_pred)
roc_auc_score(y_test, y_prob)

# model validasyonu
cv_results = cross_validate(best_cart_model_tuned,
                            X_test,
                            y_test,
                            cv=stf,
                            scoring=["f1", "roc_auc", "precision", "recall", "accuracy"])

cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()
cv_results["test_precision"].mean()
cv_results["test_recall"].mean()
cv_results["test_accuracy"].mean()

#########################
# 6. Feature Importance
#########################
best_cart_model_tuned.feature_importances_

pd.DataFrame({"feature_names": X_train.columns,
              "feature_val": best_cart_model_tuned.feature_importances_}).sort_values(by="feature_val",
                                                                                      ascending=False)

type(best_cart_model_tuned).__name__

best_cart_model_tuned.__annotations__
best_cart_model_tuned.__init__
best_cart_model_tuned.__class__
best_cart_model_tuned.__eq__
best_cart_model_tuned.__ne__()
best_cart_model_tuned.__delattr__
best_cart_model_tuned.__dict__['criterion']
best_cart_model_tuned.__format__
best_cart_model_tuned.__dir__
best_cart_model_tuned.__getattribute__
best_cart_model_tuned.__getstate__
best_cart_model_tuned.__module__
print(best_cart_model_tuned.__doc__)
best_cart_model_tuned.__str__
best_cart_model_tuned.__sizeof__
best_cart_model_tuned.__init_subclass__


def feature_imp_plot(model, features, num=5, save=False):
    imp_df = pd.DataFrame({"feature_names": features.columns,
                           "feature_val": model.feature_importances_})

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(data=imp_df.sort_values(by="feature_val", ascending=False)[:num], x="feature_val", y="feature_names")
    plt.title(
        f" model: {type(model).__name__}, criterion: {model.__dict__['criterion']}, splitter: {model.__dict__['splitter']}\n first {num} variables importance levels  ")
    plt.xlabel("importance level")
    plt.ylabel("variables name")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('imp.png')


feature_imp_plot(best_cart_model_tuned, X_train, 4)

# 7. Analyzing Model Complexity with Learning Curves (BONUS)
train_score, test_score = validation_curve(best_cart_model_tuned, X, y, param_name="max_depth", param_range=(1, 11),
                                           scoring="f1")


def val_curve_params(model, X, y, param_name, param_range, scoring, cv):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")

    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


cart_val_params = [["max_depth", range(1, 10)], ["min_samples_split", range(1, 10)]]
for i in range(len(cart_val_params)):
    val_curve_params(best_cart_model_tuned, X_train, y_train,
                     param_name=cart_val_params[i][0],
                     param_range=cart_val_params[i][1],
                     scoring="roc_auc",
                     cv=5)

# 8. Visualizing the Decision Tree
# conda install graphviz
import graphviz
import pydotplus


def tree_graph(model, col_names, file_name):
    """
    karar agacımızın görselleştirilmesi

    :param model: model nesnesi
    :param col_names: hiperparametre adımız
    :param file_name: kayıt ismi
    :return:
    """
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=best_cart_model_tuned, col_names=X.columns, file_name="finl.png")

best_cart_model_tuned.get_params()

# 9. Extracting Decision Rules
tree_rules = export_text(best_cart_model_tuned, feature_names=list(X.columns))
print(tree_rules)

# 10. Extracting Python/SQL/Excel Codes of Decision Rules
from skompiler import skompile

print(skompile(best_cart_model_tuned.predict).to('python/code'))

print(skompile(best_cart_model_tuned.predict).to('sqlalchemy/sqlite'))

print(skompile(best_cart_model_tuned.predict).to('excel'))

# 11. Prediction using Python Codes
df.sample(1)
Q = [38.0, 10863.05, 1, 5.59, 11.0, 4.0, 3.0, 3.0, 0.147105, 46.287428]


def predict_with_python_code(x):
    return (((0 if x[1] <= -1.0101985037326813 else 0) if x[3] <= -0.10843373462557793
             else 0 if x[4] <= 0.1666666716337204 else 0) if x[4] <=
                                                             0.5000000149011612 else (
        0 if x[3] <= 0.3614457845687866 else 0) if x[4
                                                   ] <= 0.8333333432674408 else 0 if x[
                                                                                         3] <= -0.39457830786705017 else 1)

predict_with_python_code(Q)
# 12. Saving and Loading Model
joblib.dump(best_cart_model_tuned, "best_cart_model_tuned.pkl")

cart_model_from_disc = joblib.load("best_cart_model_tuned.pkl")
