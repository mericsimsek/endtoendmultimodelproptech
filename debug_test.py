# Debug script - app.py'ye ekleyin veya ayrı çalıştırın

import pandas as pd
import numpy as np

# 1️⃣ İSPARK verisini kontrol et
ISPARK_DATA_PATH = "data/ispark_data.csv"

try:
    park_df = pd.read_csv(ISPARK_DATA_PATH)
    print(f"✅ İSPARK verisi yüklendi: {len(park_df)} kayıt")
    print(f"\n📋 İlk 3 satır:")
    print(park_df.head(3))
    
    # Sütun isimlerini kontrol et
    print(f"\n🔤 Sütun isimleri:")
    print(park_df.columns.tolist())
    
    # Sayısal dönüşüm
    park_df['ENLEM'] = pd.to_numeric(park_df['ENLEM'], errors='coerce')
    park_df['BOYLAM'] = pd.to_numeric(park_df['BOYLAM'], errors='coerce')
    park_df['PARK KAPASİTESİ'] = pd.to_numeric(park_df['PARK KAPASİTESİ'], errors='coerce')
    
    # Eksik veri kontrolü
    print(f"\n🔍 Eksik veri sayısı:")
    print(park_df[['ENLEM', 'BOYLAM', 'PARK KAPASİTESİ']].isnull().sum())
    
    # Temizle
    park_df_clean = park_df.dropna(subset=['ENLEM', 'BOYLAM', 'PARK KAPASİTESİ'])
    print(f"\n✅ Temiz veri: {len(park_df_clean)} kayıt")
    
    # 2️⃣ Kadıköy koordinatlarıyla test et
    test_lat = 40.9925  # Kadıköy (Mock data)
    test_lon = 29.0250
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dLat = np.radians(lat2 - lat1)
        dLon = np.radians(lon2 - lon1)
        a = np.sin(dLat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    # Mesafeleri hesapla
    park_df_clean['mesafe'] = park_df_clean.apply(
        lambda row: haversine(test_lat, test_lon, row['ENLEM'], row['BOYLAM']),
        axis=1
    )
    
    # 1 km içindekiler
    nearby = park_df_clean[park_df_clean['mesafe'] <= 1.0]
    print(f"\n🚗 Kadıköy (40.9925, 29.0250) civarında 1km içinde {len(nearby)} otopark bulundu")
    
    if len(nearby) > 0:
        print("\n📍 En yakın 5 otopark:")
        print(nearby.nsmallest(5, 'mesafe')[['OTOPARK ADI', 'PARK KAPASİTESİ', 'BULUNDUĞU İLÇE', 'mesafe']])
        
        # Skor hesapla
        weighted_score = (nearby['PARK KAPASİTESİ'] / (nearby['mesafe'] + 0.1)).sum()
        final_score = np.log1p(weighted_score / 100) * 2.5
        final_score = np.clip(final_score, 1, 10)
        print(f"\n⭐ Hesaplanan Park Skoru: {final_score:.1f}/10")
    else:
        print("\n❌ 1 km içinde hiç otopark bulunamadı!")
        print("\nEn yakın 5 otopark (mesafe fark etmeksizin):")
        print(park_df_clean.nsmallest(5, 'mesafe')[['OTOPARK ADI', 'BULUNDUĞU İLÇE', 'mesafe']])
    
except FileNotFoundError:
    print(f"❌ Dosya bulunamadı: {ISPARK_DATA_PATH}")
except Exception as e:
    print(f"❌ Hata: {e}")
    import traceback
    traceback.print_exc()