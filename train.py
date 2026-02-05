import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# 1) DATASET YÜKLE
# =========================
df = pd.read_csv("data/heart.csv")

# X ve y ayır
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 2) MODEL OLUŞTUR (TUNED RF)
# =========================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

# Modeli eğit
rf_model.fit(X_train, y_train)

# =========================
# 3) TEST PERFORMANSI
# =========================
y_pred_rf = rf_model.predict(X_test)

print("=== RANDOM FOREST TEST ===")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# =========================
# 4) TRAIN PERFORMANSI (OVERFITTING KONTROLÜ)
# =========================
y_train_pred = rf_model.predict(X_train)

print("=== RANDOM FOREST TRAIN ===")
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

# =========================
# 5) MODELİ KAYDET
# =========================
joblib.dump({"model": rf_model, "features": X.columns.tolist()}, "heart_model.pkl")

print("Model başarıyla kaydedildi: heart_model.pkl")
