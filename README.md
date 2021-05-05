# Titanic

Kaggle'ın ikonik makine öğrenmesi veri seti olan Titanic (https://www.kaggle.com/c/titanic) veri setini kullanarak çeşitli makine öğrenmesi algoritmaları ve hyperparameter tuning metodları deniyorum.

Veriyi temizlerken sayısal bakımdan yorumlanması güç olan kolonları düşürerek başladım (örneğin PassengerId kolonu gibi). Daha sonra çok fazla eksik veri olan kolon var mı diye baktım (Cabin kolonu). Ardından doldurulması mümkün olan eksikler barındıran kolonları o kolondaki ortalama değerlerle doldurdum (Age kolonu).

EDA yaparken ilk iş temizlenmiş verinin korelasyon matrisini çizdim. Ardından matristeki en yüksek korelasyon değerine sahip kolonların grafiklerini çizmeyi denedim. 
