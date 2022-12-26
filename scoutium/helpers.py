##################################################################################
# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
##################################################################################
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sc_attr_ = pd.read_csv("scoutium/scoutium_attributes.csv", sep=";")
sc_pot_ = pd.read_csv("scoutium/scoutium_potential_labels.csv", sep=";")

sc_attr = sc_attr_.copy()
sc_pot = sc_pot_.copy()

sc_pot.head()
sc_pot.shape
sc_pot.columns
sc_attr.head()
sc_attr.shape
sc_attr.columns

#############################################################################
# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id"
# 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
#############################################################################
df_merged = pd.merge(sc_pot, sc_attr, on=["task_response_id",
                                          'match_id',
                                          'evaluator_id',
                                          "player_id"])

df_merged.head()
df_merged.shape

###############################################################################
# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
###############################################################################


df_merged_kalecisiz = df_merged.loc[df_merged["position_id"] != 1]

df_merged_kalecisiz.groupby("position_id").count()
################################################################################
# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
# ( below_average sınıfı tüm verisetinin %1'ini oluşturur)
###############################################################################

df_mreged_kalecisiz_below_ave = df_merged_kalecisiz.loc[df_merged_kalecisiz["potential_label"] != "below_average"]

df = df_mreged_kalecisiz_below_ave.copy()

df.groupby("potential_label").count()

#################################################################################
# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
# Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.
#################################################################################
df.head()
pd.set_option("display.max_columns", 500)
pd.set_option("display.expand_frame_repr", False)

x = df.pivot_table(values="attribute_value",
                   index=["player_id", "position_id", "potential_label"],
                   columns="attribute_id")
x.reset_index(inplace=True)
x.columns = [str(col) for col in x.columns]
x.head()

################################################################################
# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini
# (average, highlighted) sayısal olarak ifade ediniz
################################################################################
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(x["potential_label"])
x["potential_label"] = le.transform(x["potential_label"])
x.head()

x.groupby("potential_label").count()
################################################################################
# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
################################################################################

num_cols = [col for col in x.columns if col not in ["player_id", "potential_label"]]
num_cols

################################################################################
# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek
# için StandardScaler uygulayınız.
################################################################################
from sklearn.preprocessing import StandardScaler

sScaler = StandardScaler()

sScaler.fit(x[num_cols])
x[num_cols] = sScaler.transform(x[num_cols])
x.head()

################################################################################
# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel
# etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
################################################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix, plot_confusion_matrix, roc_auc_score, roc_curve, plot_roc_curve, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.model_selection import GridSearchCV

X = x.drop(["potential_label", "player_id"], axis=1)
y = x["potential_label"]

x.groupby("potential_label").count()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

# KNN modeli
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
cv_results = cross_validate(knn, X_train, y_train, cv=5,
                            scoring=["matthews_corrcoef", "roc_auc", "f1", "precision", "recall", "accuracy"])
cv_results['test_roc_auc'].mean()
cv_results['test_f1'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_accuracy'].mean()
cv_results['test_matthews_corrcoef'].mean()
y_pred = knn.predict(X_train)
confusion_matrix(y_pred, y_train)
knn.get_params()
knn_params = {"n_neighbors": range(3, 20),
              "metric": ["minkowski", "euclidean", "manhattan"]}

knn_randomized = GridSearchCV(knn, knn_params, cv=5, scoring="roc_auc")
knn_randomized.fit(X_train, y_train)
knn_randomized.best_params_
knn_final = knn.set_params(**knn_randomized.best_params_)
knn_final.fit(X_train, y_train)
y_pred = knn_final.predict(X_test)
confusion_matrix(y_test, y_pred)

# RF modeli
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
cv_results = cross_validate(rf, X_train, y_train, cv=5,
                            scoring=["matthews_corrcoef", "roc_auc", "f1", "precision", "recall", "accuracy"])
cv_results['test_roc_auc'].mean()
cv_results['test_f1'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_accuracy'].mean()
cv_results['test_matthews_corrcoef'].mean()
y_pred = rf.predict(X_train)
confusion_matrix(y_pred, y_train)
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}
rf_random = RandomizedSearchCV(rf, rf_params, scoring="roc_auc")
rf_random.fit(X_train, y_train)
rf_random.best_params_
rf_final = RandomForestClassifier(n_estimators=500, min_samples_split=15, max_depth=8).fit(X_train, y_train)
cv_results_tuned = cross_validate(rf_final, X_train, y_train, cv=5,
                                  scoring=["matthews_corrcoef", "roc_auc", "f1", "precision", "recall", "accuracy"])
cv_results_tuned['test_roc_auc'].mean()
cv_results_tuned['test_f1'].mean()
cv_results_tuned['test_precision'].mean()
cv_results_tuned['test_recall'].mean()
cv_results_tuned['test_accuracy'].mean()
cv_results_tuned['test_matthews_corrcoef'].mean()
y_pred = rf_final.predict(X_test)
confusion_matrix(y_pred, y_test)
roc_curve(y_test, y_pred)
plot_roc_curve(rf_final, X_test, y_test)
plt.show()

# xgboost
from xgboost import XGBClassifier

xgbm = XGBClassifier()
xgbm.get_params
xgbm.fit(X_train, y_train)
cv_results = cross_validate(xgbm, X_train, y_train, cv=5,
                            scoring=["matthews_corrcoef", "roc_auc", "f1", "precision", "recall", "accuracy"])
cv_results_tuned['test_roc_auc'].mean()
cv_results_tuned['test_f1'].mean()
cv_results_tuned['test_precision'].mean()
cv_results_tuned['test_recall'].mean()
cv_results_tuned['test_accuracy'].mean()
cv_results_tuned['test_matthews_corrcoef'].mean()
y_pred = xgbm.predict(X_train)
confusion_matrix(y_train, y_pred)

xgbm_params = {"learning_rate": [0.01, 0.1],
               "max_depth": [3, 8, 10],
               "n_estimators": [100, 500, 1000],
               "subsample": [1, 0.5, 0.7]}

xgbm_random = GridSearchCV(xgbm, xgbm_params, cv=5, scoring="roc_auc")
xgbm_random.fit(X_train, y_train)
xgbm_random.best_params_

xgbm_final = xgbm.set_params(**xgbm_random.best_params_)
xgbm_final.fit(X_train, y_train)
y_pred = xgbm.predict(X_test)
confusion_matrix(y_test, y_pred)


################################################################################
# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance
# fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(xgbm_final, X_train, num=len(X_train))
plot_importance(rf_final, X_train, num=len(X_train))
plot_importance(knn_final, X_train, num=len(X_train))
