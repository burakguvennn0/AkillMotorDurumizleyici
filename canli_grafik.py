import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import random

print("Yapay Zeka ve Veriler Yüklendi...")
model = joblib.load('motor_yapay_zeka.pkl')
df = pd.read_csv("YAPAY_ZEKA_VERI_SETI.csv")

pencere = 500
toplam_veri = len(df)

plt.ion() 
fig, ax = plt.subplots(figsize=(10, 5))
fig.canvas.manager.set_window_title('Kestirimci Bakım Canlı İzleme')

print("Canlı Grafik Ekranı Başlatılıyor... (Kapatmak için penceredeki X tuşuna bas)")

while plt.fignum_exists(fig.number):
    rastgele_an = random.randint(0, toplam_veri - pencere)
    
    paket = df['Titresim'].iloc[rastgele_an : rastgele_an+pencere].values
    gercek_durum = df['Durum'].iloc[rastgele_an]
    
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
    
    ax.clear() 
    ax.plot(paket, color='royalblue', linewidth=1.5)
    ax.set_ylim(-2.5, 2.5) 
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if tahmin_edilen == "Normal_Calisma":
        baslik = f"DURUM: MOTOR SAĞLIKLI | RMS: {rms:.3f}"
        ax.set_title(baslik, color='forestgreen', fontsize=16, fontweight='bold')
        ax.set_facecolor('#f0fff0')
    else:
        baslik = f"DİKKAT: {tahmin_edilen.upper()} | RMS: {rms:.3f}"
        ax.set_title(baslik, color='crimson', fontsize=16, fontweight='bold')
        ax.set_facecolor('#ffeeee')
        
    ax.set_xlabel("Zaman (Örneklem)")
    ax.set_ylabel("Titreşim Genliği")
    
    plt.pause(0.5)

plt.ioff()
