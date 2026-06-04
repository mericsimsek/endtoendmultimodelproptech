#!/usr/bin/env python3
"""
Offline Park Yeri Analiz Scripti
Bu script Flask'tan bağımsız çalışır ve static/parking_data/ klasöründeki
tüm ilçe klasörlerini tarayarak YOLO ile analiz yapar.
MongoDB'ye de sonuçları kaydeder.
"""

import os
import json
import torch
import cv2
from pathlib import Path
from datetime import datetime
import logging
import sys

# Database import
try:
    from database import MongoDatabaseManager
    DB_AVAILABLE = True
except ImportError:
    print("⚠️ database.py bulunamadı, sadece JSON'a kayıt yapılacak")
    DB_AVAILABLE = False

# Loglama ayarı
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParkingImageAnalyzer:
    def __init__(self):
        self.base_path = Path("static/parking_data")
        self.model_path = "models/yolov5/yolov5/runs/train/exp5/weights/best.pt"
        self.yolo_model = None
        self.load_model()
    
    def load_model(self):
        """YOLO modelini yükle"""
        try:
            if os.path.exists(self.model_path):
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                self.yolo_model.eval()
                logger.info("✅ YOLO modeli başarıyla yüklendi")
            else:
                logger.error(f"❌ Model dosyası bulunamadı: {self.model_path}")
                raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            raise
    
    def get_district_folders(self):
        """static/parking_data/ altındaki tüm ilçe klasörlerini getir"""
        if not self.base_path.exists():
            logger.warning(f"⚠️ Ana klasör bulunamadı: {self.base_path}")
            self.base_path.mkdir(parents=True, exist_ok=True)
            return []
        
        districts = [d for d in self.base_path.iterdir() if d.is_dir()]
        logger.info(f"📁 {len(districts)} ilçe klasörü bulundu")
        return districts
    
    def analyze_district(self, district_path):
        """Bir ilçe klasörünü analiz et"""
        district_name = district_path.name
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 {district_name} ilçesi analiz ediliyor...")
        logger.info(f"{'='*60}")
        
        images_path = district_path / "images"
        results_path = district_path / "results"
        results_json_path = district_path / "results.json"
        
        # Klasörleri oluştur
        images_path.mkdir(exist_ok=True)
        results_path.mkdir(exist_ok=True)
        
        # Mevcut results.json'ı oku veya yeni oluştur
        if results_json_path.exists():
            with open(results_json_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
        else:
            results_data = {
                "summary": {
                    "park_convenience_score": 0,
                    "total_capacity": 0,
                    "average_empty": 0,
                    "average_occupied": 0
                },
                "images": []
            }
        
        # Daha önce analiz edilmiş resimleri bul
        analyzed_images = {img['original_path'] for img in results_data['images']}
        
        # Yeni resimleri bul
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")) + list(images_path.glob("*.jpeg"))
        new_images = []
        
        for img_file in image_files:
            relative_path = f"static/parking_data/{district_name}/images/{img_file.name}"
            if relative_path not in analyzed_images:
                new_images.append(img_file)
        
        if not new_images:
            logger.info(f"ℹ️ {district_name} için yeni resim bulunamadı (Toplam: {len(image_files)})")
            return
        
        logger.info(f"🆕 {len(new_images)} yeni resim analiz edilecek")
        
        # Her yeni resmi analiz et
        for img_file in new_images:
            self.analyze_single_image(img_file, district_name, results_path, results_data)
        
        # Özet istatistikleri güncelle
        self.update_summary(results_data)
        
        # results.json'ı kaydet
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ {district_name} ilçesi analizi tamamlandı!")
        logger.info(f"📊 Park Kolaylığı Skoru: {results_data['summary']['park_convenience_score']}/10")
    
    def analyze_single_image(self, img_path, district_name, results_path, results_data):
        """Tek bir resmi YOLO ile analiz et"""
        img_name = img_path.name
        logger.info(f"  📸 Analiz ediliyor: {img_name}")
        
        try:
            # YOLO ile analiz yap
            results = self.yolo_model(str(img_path))
            
            # Sonuç resmini kaydet
            result_img_name = f"{img_path.stem}_result{img_path.suffix}"
            result_img_path = results_path / result_img_name
            import cv2 as _cv2
            rendered = results.render()[0]  # RGB numpy array
            _cv2.imwrite(str(result_img_path),
                         _cv2.cvtColor(rendered, _cv2.COLOR_RGB2BGR))
            
            # Dolu ve Boş sayılarını hesapla
            detections = results.pandas().xyxy[0]
            occupied_spots = len(detections[detections['name'] == 'Dolu'])
            empty_spots = len(detections[detections['name'] == 'Bos'])
            total_spots = occupied_spots + empty_spots
            
            # Sonucu results.json'a ekle
            image_result = {
                "original_path": f"static/parking_data/{district_name}/images/{img_name}",
                "result_path": f"static/parking_data/{district_name}/results/{result_img_name}",
                "occupied_spots": occupied_spots,
                "empty_spots": empty_spots,
                "total_spots": total_spots,
                "analyzed_at": datetime.now().isoformat()
            }
            
            results_data['images'].append(image_result)
            
            logger.info(f"    ✓ Dolu: {occupied_spots} | Boş: {empty_spots} | Toplam: {total_spots}")
            
        except Exception as e:
            logger.error(f"    ❌ {img_name} analiz hatası: {e}")
    
    def update_summary(self, results_data):
        """Özet istatistikleri güncelle"""
        if not results_data['images']:
            return
        
        total_occupied = sum(img['occupied_spots'] for img in results_data['images'])
        total_empty = sum(img['empty_spots'] for img in results_data['images'])
        total_capacity = total_occupied + total_empty
        
        # Park Kolaylığı Skoru hesapla (boş yer oranı * 10)
        if total_capacity > 0:
            score = (total_empty / total_capacity) * 10
        else:
            score = 0
        
        results_data['summary'] = {
            "park_convenience_score": round(score, 1),
            "total_capacity": total_capacity,
            "average_empty": total_empty,
            "average_occupied": total_occupied,
            "last_updated": datetime.now().isoformat()
        }
    
    def run(self):
        """Ana analiz döngüsü"""
        logger.info("\n" + "="*60)
        logger.info("🚀 OFFLINE PARK YERİ ANALİZ SİSTEMİ BAŞLATILDI")
        logger.info("="*60 + "\n")
        
        if not self.yolo_model:
            logger.error("❌ YOLO modeli yüklü değil, çıkılıyor...")
            return
        
        districts = self.get_district_folders()
        
        if not districts:
            logger.warning("⚠️ Analiz edilecek ilçe bulunamadı!")
            logger.info("\n💡 Kullanım: static/parking_data/{ilçe_adı}/images/ klasörüne resimler koyun")
            return
        
        # Her ilçeyi analiz et
        for district_path in districts:
            try:
                self.analyze_district(district_path)
            except Exception as e:
                logger.error(f"❌ {district_path.name} analizi başarısız: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ TÜM ANALİZLER TAMAMLANDI!")
        logger.info("="*60)


if __name__ == "__main__":
    analyzer = ParkingImageAnalyzer()
    analyzer.run()