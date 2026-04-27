import os
import scipy.io as sio
import pandas as pd

print("Laboratuvar verileri taranıyor, lütfen bekle...\n")

tum_veriler = []

for dosya_adi in os.listdir('.'):
    if dosya_adi.endswith('.mat'):
        print(f"-> İşleniyor: {dosya_adi}")
        mat_icerik = sio.loadmat(dosya_adi)

        titresim_verisi = None
        for baslik, veri in mat_icerik.items():
            if "DE_time" in baslik:
                titresim_verisi = veri.flatten() 
                break

        if titresim_verisi is None:
             for baslik, veri in mat_icerik.items():
                if not baslik.startswith('__') and len(veri) > 1000:
                    titresim_verisi = veri.flatten()
                    break

        if titresim_verisi is not None:
            if "Normal" in dosya_adi:
                etiket = 0
                durum = "Normal_Calisma"
            elif "B0" in dosya_adi:
                etiket = 1
                durum = "Bilya_Hasari"
            elif "IR" in dosya_adi:
                etiket = 2
                durum = "Ic_Bilezik_Hasari"
            elif "OR" in dosya_adi:
                etiket = 3
                durum = "Dis_Bilezik_Hasari"
            else:
                etiket = 99
                durum = "Bilinmeyen"

            df = pd.DataFrame({
                "Titresim": titresim_verisi,
                "Etiket": etiket,
                "Durum": durum
            })
            tum_veriler.append(df)

if tum_veriler:
    son_tablo = pd.concat(tum_veriler, ignore_index=True)
    son_tablo.to_csv("YAPAY_ZEKA_VERI_SETI.csv", index=False)
    print("\nMÜKEMMEL! İşlem tamamlandı.")
    print("Klasörüne bak: 'YAPAY_ZEKA_VERI_SETI.csv' adında makine öğrenmesine hazır dosyan oluşturuldu.")
else:
    print("\nHATA: Klasörde uygun veri bulunamadı.")