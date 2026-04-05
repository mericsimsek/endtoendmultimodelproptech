import random
import numpy as np
from database import MongoDatabaseManager

# Veritabanı bağlantısını başlat
db = MongoDatabaseManager()

# Temel alacağımız, yeni özelliklere sahip ve temizlenmiş 10 ilan
# Not: 'rooms', 'total_rooms' gibi eski anahtarlar temizlendi, sadece 'totalrooms' kullanılıyor.
base_properties = [
    # Kadıköy - Orta-Üst Bölge
    {
        'title': 'Geniş Daire - Kadıköy', 'price': 8500000, 'm2': 120, 'totalrooms': 3, 'bathroom_count': 2,
        'province': 'İstanbul', 'district': 'Kadıköy', 'neighborhood': 'Moda',
        'location_score': 8.0, 'luxury_score': 7.5
    },
    # Bağcılar - Orta-Alt Bölge
    {
        'title': 'Ekonomik Daire - Bağcılar', 'price': 2200000, 'm2': 70, 'totalrooms': 1, 'bathroom_count': 1,
        'province': 'İstanbul', 'district': 'Bağcılar', 'neighborhood': 'Merkez',
        'location_score': 4.5, 'luxury_score': 4.0
    },
    # Beşiktaş - Premium Bölge
    {
        'title': 'Lüks Daire - Beşiktaş', 'price': 15000000, 'm2': 140, 'totalrooms': 3, 'bathroom_count': 2,
        'province': 'İstanbul', 'district': 'Beşiktaş', 'neighborhood': 'Bebek',
        'location_score': 9.5, 'luxury_score': 9.0
    },
    # Esenyurt - Ekonomik Bölge
    {
        'title': 'Fırsat Dairesi - Esenyurt', 'price': 1800000, 'm2': 60, 'totalrooms': 1, 'bathroom_count': 1,
        'province': 'İstanbul', 'district': 'Esenyurt', 'neighborhood': 'Merkez',
        'location_score': 3.5, 'luxury_score': 3.5
    },
    # Şişli - Orta-Üst Bölge
    {
        'title': 'İş Merkezi Yakını - Şişli', 'price': 7200000, 'm2': 80, 'totalrooms': 2, 'bathroom_count': 1,
        'province': 'İstanbul', 'district': 'Şişli', 'neighborhood': 'Mecidiyeköy',
        'location_score': 8.5, 'luxury_score': 7.0
    },
    # Diğer 5 ilan
    {
        'title': 'Modern Daire - Kadıköy', 'price': 5500000, 'm2': 90, 'totalrooms': 2, 'bathroom_count': 1,
        'province': 'İstanbul', 'district': 'Kadıköy', 'neighborhood': 'Hasanpaşa',
        'location_score': 7.5, 'luxury_score': 6.5
    },
    {
        'title': 'Aileniz İçin - Bağcılar', 'price': 3200000, 'm2': 88, 'totalrooms': 2, 'bathroom_count': 1,
        'province': 'İstanbul', 'district': 'Bağcılar', 'neighborhood': 'Yenimahalle',
        'location_score': 5.0, 'luxury_score': 5.0
    },
    {
        'title': 'Butik Daire - Beşiktaş', 'price': 9500000, 'm2': 90, 'totalrooms': 2, 'bathroom_count': 1,
        'province': 'İstanbul', 'district': 'Beşiktaş', 'neighborhood': 'Levent',
        'location_score': 9.0, 'luxury_score': 8.5
    },
    {
        'title': 'Geniş Aile Evi - Esenyurt', 'price': 2800000, 'm2': 105, 'totalrooms': 3, 'bathroom_count': 1,
        'province': 'İstanbul', 'district': 'Esenyurt', 'neighborhood': 'Kıraç',
        'location_score': 4.0, 'luxury_score': 4.5
    },
    {
        'title': 'Dubleks Villa - Şişli', 'price': 12500000, 'm2': 160, 'totalrooms': 4, 'bathroom_count': 3,
        'province': 'İstanbul', 'district': 'Şişli', 'neighborhood': 'Nişantaşı',
        'location_score': 9.0, 'luxury_score': 8.5
    }
]

NUM_VARIATIONS_PER_PROPERTY = 15
generated_properties = []

print(f"{len(base_properties)} temel ilan üzerinden {len(base_properties) * NUM_VARIATIONS_PER_PROPERTY} adet yeni ilan üretiliyor...")

for base_prop in base_properties:
    for i in range(NUM_VARIATIONS_PER_PROPERTY):
        new_prop = base_prop.copy()
        
        # --- 1. Temel Sayısal Değerleri Rastgele Değiştir ---
        new_prop['m2'] = max(20, int(base_prop['m2'] * (1 + random.uniform(-0.5, 0.5))))
        new_prop['price'] = max(500000, int(base_prop['price'] * (1 + random.uniform(-0.5, 0.5))))
        
        new_location_score = base_prop['location_score'] * (1 + random.uniform(-0.1, 0.1))
        new_luxury_score = base_prop['luxury_score'] * (1 + random.uniform(-0.1, 0.1))
        new_prop['location_score'] = round(max(1.0, min(10.0, new_location_score)), 1)
        new_prop['luxury_score'] = round(max(1.0, min(10.0, new_luxury_score)), 1)
        new_prop['luxury_score_final'] = new_prop['luxury_score']

        # --- 2. YENİ ÖZELLİKLERİ EKLE ve HESAPLA ---
        
        # Rastgele olarak 'sale' (satılık) veya 'rent' (kiralık) ata
        new_prop['listing_type'] = random.choice(['sale', 'rent'])
        
        # Kiralama fiyatını, YENİ satış fiyatına göre %0.4 ile %0.6 arasında rastgele bir oranla hesapla
        # ve daha gerçekçi olması için en yakın 100'e yuvarla (örn: 42,123 -> 42,100)
        new_rent_price = new_prop['price'] * random.uniform(0.004, 0.006)
        new_prop['rent_price'] = int(round(new_rent_price / 100) * 100)

        # --- 3. Türetilmiş Değerleri Yeniden Hesapla ---
        
        if new_prop['m2'] > 0:
            new_prop['price_per_m2'] = int(new_prop['price'] / new_prop['m2'])
            new_prop['room_density'] = new_prop['totalrooms'] / new_prop['m2']
        else:
            new_prop['price_per_m2'] = new_prop['price']
            new_prop['room_density'] = 0

        new_prop['location_tier'] = int((new_prop['location_score'] - 1) / 2)
        new_prop['m2_location_interact'] = new_prop['m2'] * np.sqrt(new_prop['location_score'])
        
        # --- 4. Diğer Özellikleri Güncelle ---
        
        listing_text = "Kiralık" if new_prop['listing_type'] == 'rent' else "Satılık"
        new_prop['title'] = f"{listing_text} {new_prop['totalrooms']} Odalı Daire - {new_prop['district']}"
        new_prop['description'] = f"{new_prop['neighborhood']} mahallesinde, {new_prop['m2']} metrekare genişliğinde fırsat."
        
        # Kiralık ilanların eşyalı olma ihtimalini artıralım
        if new_prop['listing_type'] == 'rent':
            new_prop['furnished'] = random.choice([True, True, False]) # %66 ihtimalle eşyalı
        else:
            new_prop['furnished'] = random.choice([True, False]) # %50 ihtimalle eşyalı
            
        new_prop['air_conditioning'] = random.choice([True, False])
        new_prop['has_balcony'] = random.choice([True, False])
        new_prop['has_elevator'] = random.choice([True, False])
        new_prop['has_parking'] = random.choice([True, False])
        
        # Eski ve gereksiz anahtarları temizle
        new_prop.pop('description', None) # Base'den gelen eski description'ı sil
        new_prop.pop('price_per_m2', None) # Base'den gelen eski price_per_m2'yi sil
        
        generated_properties.append(new_prop)

print(f"✅ {len(generated_properties)} adet yeni ilan başarıyla üretildi.")
print("\nVeritabanına ekleme işlemi başlıyor...")

# Üretilen tüm ilanları veritabanına ekle
for prop in generated_properties:
    prop_id = db.insert_property(prop)
    if prop_id:
        print(f"✓ Eklendi: {prop['title']} (ID: {prop_id})")
    else:
        print(f"✗ Hata: {prop['title']} eklenirken bir sorun oluştu.")

print(f"\n✨ Toplam {len(generated_properties)} yeni ilan veritabanına eklendi.")