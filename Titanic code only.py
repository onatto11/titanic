import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

train = pd.read_csv("I:/train.csv")
test = pd.read_csv("I:/test.csv")

# # Veri temizliği

#Yaştaki NaN değerleri veri setinin yaş ortalamasıyla doldurmak ideal gibi
#Kabin kolonunu düşürmek gerek, herhangi bir imputation metodu uygulamak için çok az kullanılabilir veri var
#Embarked kolonundaki iki NaN satırı düşürmek gerek
train.info()

train.head(15)

train.Embarked.value_counts()

train.dropna(subset=["Embarked"], inplace=True)
train.drop("Cabin", axis=1, inplace=True)

#Yaş sütunu hariç her sütunda 889 satır var, Age sütununu da ortalamalarla doldurunca 889 olacak
train.info()

#Yaş sütununu da ortalama yaşı kullanarak doldurdum
mean_age = train.Age.mean()
train.fillna(value=mean_age, inplace=True)
train.info()

#Object veri tiplerini int64'e çevirerek matematiksel işlem yapılabilecek hale getirelim
train.replace({"male":0, "female":1}, inplace = True)
train.replace({"S":0, "C":1, "Q":2}, inplace = True)

#Yolcu isimleri ve bilet kodları algoritma tarafından kullanılabilecek sayısal girdilere çevrilemez, bu yüzden düşürülmesi gerek.
#Ayrıca PassengerId sütununun da her ne kadar int64 olsa da yorumlanabilir bir sayısal değer olmamasından ötürü düşürülmesi şart.
train.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

#Veri temiz ve işlenmeye hazır duruyor.
train.info()
train.columns

# # Feature engineering

#Kadınlar ve çocuklar daha fazla hayatta kaldığından 
#yaş ortalamasının altındaki kadınları gösteren bir kolon
#belki bir şeyler anlatabilir.
train["Young_Female"] = train[(train["Age"] <= train.Age.mean()) & (train["Sex"] == 1)].Age

# # Veri görselleştirme

sns.set_context("talk")
ax = plt.subplots(figsize=(15, 10))
sns.heatmap(train.corr(), annot=True)
plt.title("Kolonların korelasyon matrisi")

sns.set_context("talk")
g = sns.countplot("Survived", data=train)
plt.title("Hayatta kalanların sayısı")
g.set_xticklabels(["Dead", "Alive"])

sns.set_context("talk")
ax = plt.subplots(figsize=(15, 10))
g1 = sns.boxenplot("Pclass", "Fare", data=train)
g1.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
plt.title("Yolcu sınıfına göre bilet ücretlerinin dağılım grafiği")
plt.show()

sns.set_context("talk")
g = sns.catplot("Pclass", "Survived", data=train, kind="bar", ci=None)
g.set_yticklabels(["%0", "%10", "%20", "%30", "%40", "%50", "%60"])
g.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
plt.title("Yolcu sınıfına göre yolcuların ortalama hayatta kalma yüzdeleri")

sns.set_context("talk")
g = sns.catplot("Sex", "Survived", data=train, kind="bar", ci=None)
g.set_yticklabels(["%0", "%10", "%20", "%30", "%40", "%50", "%60", "%70"])
g.set_xticklabels(["Male", "Female"])
plt.title("Cinsiyete göre yolcuların ortalama hayatta kalma yüzdeleri")

sns.set_context("talk")
ax = plt.subplots(figsize=(15, 10))
a = sns.boxenplot("Survived", "Age", data=train)
a.set_xticklabels(["Dead", "Alive"])
plt.title("Hayatta kalıp kalmamalarına göre yolcuların yaş dağılımları")

train[(train["Sex"] == 1) & (train["Survived"] == 1)].Age.mean()


# # Algoritmanın kurulması

#Veri setini X ve y şeklinde ayırarak tahmin edilecek değer olan Survived kolonunu ayıralım
X = train[["Fare", "Pclass", "Sex", "Age"]]
y = np.array(train.Survived)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.3)
scaler = StandardScaler().fit(X_train)
standard_x = scaler.transform(X_train)
standard_xtest = scaler.transform(X_test)

knn = KNeighborsClassifier()
params = {"n_neighbors":np.arange(1, 31)}
grid = GridSearchCV(estimator = knn, param_grid = params, cv=5)
grid.fit(standard_x, y_train)
knn_pred = grid.predict(standard_xtest)
print(classification_report(y_test, knn_pred))

ax = plt.subplots(figsize=(12, 9))
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, xticklabels=["Dead", "Alive"], yticklabels=["Dead", "Alive"], fmt="g")
plt.ylabel("Tahminler")
plt.xlabel("Gerçek değerler")
plt.title("k-NN Algoritmasının Sınıflandırma Performansı")

print("En yüksek puan:", grid.best_score_)
print("En iyi n_neighbors parametresi:",grid.best_estimator_.n_neighbors)
print("Ortalama model puanı:", grid.cv_results_["mean_test_score"].mean())

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=3)
params_log = {"estimator__C":[0.0001, 0.001, 0.01, 0.1, 1, 1.2, 1.5, 1.8, 2, 2.2, 2.5, 2.7, 2.8, 2.81, 2.82, 2.83, 2.84, 2.85, 2.86, 2.9, 3, 3.1, 3.3, 3.5, 3.8, 10, 100]}
grid_log = GridSearchCV(estimator=rfe, param_grid = params_log, cv=5)
grid_log.fit(standard_x, y_train)
logreg_pred = grid_log.predict(standard_xtest)
print(classification_report(y_test, logreg_pred))

ax = plt.subplots(figsize=(12, 9))
sns.heatmap(confusion_matrix(y_test, logreg_pred), annot=True, xticklabels=["Dead", "Alive"], yticklabels=["Dead", "Alive"], fmt="g")
plt.ylabel("Tahminler")
plt.xlabel("Gerçek değerler")
plt.title("Lojistik Regresyon Algoritmasının Sınıflandırma Performansı")

print("En yüksek puan:", grid_log.best_score_)
print("En iyi C parametresi:", grid_log.best_estimator_.estimator_)
print("Ortalama model puanı:", grid_log.cv_results_["mean_test_score"].mean())

knn_prob = grid.predict_proba(standard_xtest)[:, 1]
logreg_prob = grid_log.predict_proba(standard_xtest)[:, 1]
fprlog, tprlog, thresholdslog = roc_curve(y_test, logreg_prob)
fpr, tpr, thresholds = roc_curve(y_test, knn_prob)
sns.set_style("darkgrid")
sns.set_context("poster")
ax = plt.subplots(figsize=(15, 10))
plt.plot([0, 1], [0, 1], 'k--')
sns.lineplot(fpr, tpr, alpha=0.3, ci=None)
sns.lineplot(fprlog, tprlog, alpha=0.3, ci=None)
plt.xlabel('Yalancı pozitif oranı')
plt.ylabel('Gerçek pozitif oranı')
plt.title('ROC eğrisi')
plt.legend(["Baz Çizgisi", "k-NN", "Lojistik Regresyon"])
plt.show()