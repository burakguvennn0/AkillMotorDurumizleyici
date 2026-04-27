import pandas as pd
import numpy as np

print("Devasa veri seti yükleniyor, bu işlem bilgisayarının hızına göre 10-20 saniye sürebilir...")
df = pd.read_csv("YAPAY_ZEKA_VERI_SETI.csv")

pencere_boyutu = 500
yeni_veriler = []

print("Veriler paketleniyor ve mühendislik hesaplamaları yapılıyor...")

for etiket in df['Etiket'].unique():
    durum_verisi = df[df['Etiket'] == etiket]['Titresim'].values
    durum_adi = df[df['Etiket'] == etiket]['Durum'].iloc[0]
    
    for i in range(0, len(durum_verisi) - pencere_boyutu, pencere_boyutu):
        paket = durum_verisi[i:i+pencere_boyutu]
        
        ortalama = np.mean(paket)
        std_sapma = np.std(paket)
        maksimum = np.max(paket)
        minimum = np.min(paket)
        rms = np.sqrt(np.mean(paket**2))
        
        yeni_veriler.append({
            'Ortalama': ortalama,
            'Standart_Sapma': std_sapma,
            'Maksimum': maksimum,
            'Minimum': minimum,
            'RMS': rms,
            'Etiket': etiket,
            'Durum': durum_adi
        })

ozellik_verisi = pd.DataFrame(yeni_veriler)
ozellik_verisi.to_csv("MODEL_EGITIM_VERISI.csv", index=False)

print("\nMÜKEMMEL! Veriler başarıyla akıllı paketlere dönüştürüldü.")
print(f"Yapay zekayı eğitmek için toplam {len(ozellik_verisi)} adet mükemmel kalitede örneğimiz oldu.")
print("Klasörüne bak: 'MODEL_EGITIM_VERISI.csv' dosyası hazır!")