import pandas as pd
import numpy as np
import joblib
import time
import os

def ekran_temizle():
    os.system('cls' if os.name == 'nt' else 'clear')

print("Yapay Zeka beyni yukleniyor...")
model = joblib.load('motor_yapay_zeka.pkl')

print("Ham veriler sensorden (dosyadan) cekiliyor...")
df = pd.read_csv("YAPAY_ZEKA_VERI_SETI.csv")

pencere = 500
toplam_veri = len(df)

print("\n--- CANLI IZLEME BASLIYOR ---\n")
time.sleep(2)

for i in range(0, toplam_veri - pencere, pencere):
    paket = df['Titresim'].iloc[i:i+pencere].values
    gercek_durum = df['Durum'].iloc[i]
    
    ortalama = np.mean(paket)
    std_sapma = np.std(paket)
    maksimum = np.max(paket)
    minimum = np.min(paket)
    rms = np.sqrt(np.mean(paket**2))
    
    anlik_veri = pd.DataFrame([{
        'Ortalama': ortalama, 
        'Standart_Sapma': std_sapma, 
        'Maksimum': maksimum, 
        'Minimum': minimum, 
        'RMS': rms
    }])
    
    tahmin_edilen = model.predict(anlik_veri)[0]
    
    ekran_temizle()
    print("="*50)
    print(" *** KESTIRIMCI BAKIM PANELI (CANLI YAYIN) ***")
    print("="*50)
    print(f"[{i} - {i+500} Arasi Titresim Verisi Isleniyor...]\n")
    print(f"-> Anlik RMS Degeri    : {rms:.4f}")
    print(f"-> Maksimum Vuruntu    : {maksimum:.4f}\n")
    
    if tahmin_edilen == "Normal_Calisma":
        print(">>> YAPAY ZEKA KARARI : MOTOR SAGLIKLI, SORUN YOK.")
    else:
        print(f"!!! YAPAY ZEKA UYARISI : {tahmin_edilen.upper()} TESPIT EDILDI !!!")
        
    print(f"\n(Sistemdeki Gercek Kayit: {gercek_durum})")
    print("="*50)
    
    time.sleep(0.5)