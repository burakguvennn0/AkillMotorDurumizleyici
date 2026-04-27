import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Yapay zeka uyandırılıyor ve eğitim verileri okunuyor...")

df = pd.read_csv("MODEL_EGITIM_VERISI.csv")

X = df[['Ortalama', 'Standart_Sapma', 'Maksimum', 'Minimum', 'RMS']]
y = df['Durum'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Random Forest (Rastgele Orman) yapay zeka modeli eğitiliyor...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nEğitim bitti! Model hiç görmediği verilerle test ediliyor...")
tahminler = model.predict(X_test)

basari_orani = accuracy_score(y_test, tahminler)
print(f"\n>>> YAPAY ZEKA BAŞARI ORANI: % {basari_orani * 100:.2f} <<<")

print("\nDetaylı Karne (Hangi arızayı ne kadar doğru bildi):")
print(classification_report(y_test, tahminler))

joblib.dump(model, 'motor_yapay_zeka.pkl')
print("\nHarika! Modelin beyni 'motor_yapay_zeka.pkl' adıyla klasörüne kaydedildi.")