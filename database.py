# database.py - KİRA DESTEĞİ VE PARK ANALİZİ EKLENMİŞ VERSİYON
from pymongo import MongoClient, ASCENDING, DESCENDING
from typing import Dict, List, Optional
import os
import logging
from datetime import datetime, timedelta, timezone
from bson import ObjectId

class MongoDatabaseManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        uri = os.getenv('MONGO_URI')
        if uri:
            self.client = MongoClient(uri)
            dbname = os.getenv('DB_NAME', 'smartcity_db1')
            self.db = self.client[dbname]
        else:
            host = os.getenv('DB_HOST', 'localhost')
            port = int(os.getenv('DB_PORT', 27017))
            user = os.getenv('DB_USER')
            pwd = os.getenv('DB_PASSWORD')
            if user and pwd:
                self.client = MongoClient(host=host, port=port, username=user, password=pwd)
            else:
                self.client = MongoClient(host=host, port=port)
            self.db = self.client[os.getenv('DB_NAME', 'smartcity_db1')]

        # collection references
        self.properties = self.db.properties
        self.parking_analyses = self.db.parking_analyses
        self.district_parking_analyses = self.db.district_parking_analyses
        self.price_predictions = self.db.price_predictions
        self.user_interactions = self.db.user_interactions
        self.metrics = self.db.metrics
        
        # location tier mapping
        self.location_tier_map = {
            0: "Ekonomik Bölge (Esenyurt, Küçükçekmece, Sultangazi civarı)",
            1: "Orta-Alt Bölge (Bahçelievler, Bağcılar, Gaziosmanpaşa civarı)",
            2: "Orta Bölge (Kadıköy, Üsküdar, Kartal civarı)",
            3: "Orta-Üst Bölge (Şişli, Sarıyer, Beykoz civarı)",
            4: "Premium Bölge (Beşiktaş, Bebek, Etiler civarı)"
        }
    
    def get_all_properties(self, limit=50):
        try:
            properties = self.properties.find({}).limit(limit)
            return list(properties)
        except Exception as e:
            self.logger.error(f"Tüm ilanlar alınırken hata oluştu: {e}")
            return []

    # ---------- helpers ----------
    def _serialize_doc(self, doc: Dict) -> Optional[Dict]:
        if not doc: return None
        out = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId): out[k] = str(v)
            elif isinstance(v, datetime): out[k] = v.isoformat()
            else: out[k] = v
        return out

    # ---------- setup ----------
    def create_collections_and_indexes(self):
        try:
            if 'properties' not in self.db.list_collection_names():
                self.db.create_collection('properties')

            self.properties.create_index([('location_tier', ASCENDING)], name='idx_location_tier')
            self.properties.create_index([('price', ASCENDING)], name='idx_price')
            self.properties.create_index([('m2', ASCENDING)], name='idx_m2')
            self.properties.create_index([('is_active', ASCENDING)], name='idx_active')
            self.properties.create_index([('created_at', DESCENDING)], name='idx_created_at')
            self.properties.create_index([('listing_type', ASCENDING)], name='idx_listing_type') 

            try:
                self.properties.create_index([('title', 'text'), ('description', 'text')],
                                             name='idx_text', default_language='turkish')
            except Exception:
                pass

            self.parking_analyses.create_index([('property_id', ASCENDING)], name='idx_parking_property')
            self.price_predictions.create_index([('property_id', ASCENDING), ('created_at', DESCENDING)],
                                                name='idx_pred_property_time')
            self.user_interactions.create_index([('property_id', ASCENDING), ('created_at', DESCENDING)],
                                                name='idx_ui_property_time')
            self.metrics.create_index([('name', ASCENDING), ('created_at', DESCENDING)], name='idx_metrics')

            self.logger.info("Collections and indexes created/ensured.")
            self.create_parking_collections_indexes()
        except Exception as e:
            self.logger.error(f"create_collections_and_indexes error: {e}")

    def create_parking_collections_indexes(self):
        try:
            if 'district_parking_analyses' not in self.db.list_collection_names():
                self.db.create_collection('district_parking_analyses')
            
            self.district_parking_analyses.create_index([('district', ASCENDING)], name='idx_district', unique=True)
            self.district_parking_analyses.create_index([('updated_at', DESCENDING)], name='idx_updated_at')
            self.district_parking_analyses.create_index([('summary.park_convenience_score', DESCENDING)], name='idx_score')
        except Exception as e:
            self.logger.error(f"Park analizi collection oluşturma hatası: {e}")

    # ---------- properties ----------
    def insert_property(self, property_data: Dict) -> Optional[str]:
        doc = property_data.copy()
        doc.setdefault('title', 'Satılık Daire')
        doc.setdefault('description', None)
        doc.setdefault('price', float(doc.get('price', 0) or 0))
        doc.setdefault('price_per_m2', doc.get('price_per_m2'))
        doc.setdefault('m2', float(doc.get('m2', 0) or 0))
        doc.setdefault('rooms', int(doc.get('rooms', 0) or 0))
        doc['total_rooms'] = int(doc.get('total_rooms', doc.get('totalrooms', 0) or 0))
        doc.setdefault('bathroom_count', int(doc.get('bathroom_count', 1)))
        doc.setdefault('province', doc.get('province', 'İstanbul'))
        doc.setdefault('district', doc.get('district'))
        doc.setdefault('neighborhood', doc.get('neighborhood'))
        doc.setdefault('location_tier', int(doc.get('location_tier', 3)))
        doc.setdefault('luxury_score', float(doc.get('luxury_score', 5)))
        doc.setdefault('room_density', doc.get('room_density'))
        doc.setdefault('m2_location_interact', doc.get('m2_location_interact'))
        doc.setdefault('furnished', bool(doc.get('furnished', False)))
        doc.setdefault('air_conditioning', bool(doc.get('air_conditioning', False)))
        doc.setdefault('has_balcony', bool(doc.get('has_balcony', False)))
        doc.setdefault('has_elevator', bool(doc.get('has_elevator', False)))
        doc.setdefault('has_parking', bool(doc.get('has_parking', False)))
        doc['is_active'] = True

        doc.setdefault('listing_type', 'sale')
        doc.setdefault('rent_price', float(doc.get('rent_price', 0) or 0))
        
        if doc['listing_type'] == 'rent' and doc['rent_price'] == 0:
            doc['rent_price'] = float(doc['price'] * 0.005)
        if doc['listing_type'] == 'sale' and doc['rent_price'] == 0:
            doc['rent_price'] = float(doc['price'] * 0.004)
        if doc['rent_price'] > 0 and doc['rent_price'] < 3000:
            doc['rent_price'] = 3000.0

        lt = doc.get('location_tier', 3)
        doc['location_description'] = self.location_tier_map.get(lt, "Bilinmeyen Bölge")

        now = datetime.now(timezone.utc)
        doc['created_at'] = now
        doc['updated_at'] = now

        try:
            res = self.properties.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            self.logger.error(f"insert_property error: {e}")
            return None

    def get_properties(self, filters: Dict = None, limit: int = 50, skip: int = 0) -> List[Dict]:
        query = {'is_active': True}
        if filters:
            if 'district' in filters: query['district'] = {'$regex': filters['district'], '$options': 'i'}
            if 'rooms' in filters: query['total_rooms'] = int(filters['rooms'])
            if 'listing_type' in filters: query['listing_type'] = filters['listing_type']
            
            if 'min_price' in filters or 'max_price' in filters:
                price_q = {}
                if 'min_price' in filters: price_q['$gte'] = float(filters['min_price'])
                if 'max_price' in filters: price_q['$lte'] = float(filters['max_price'])
                query['price'] = price_q
            
            if 'min_m2' in filters or 'max_m2' in filters:
                m2_q = {}
                if 'min_m2' in filters: m2_q['$gte'] = float(filters['min_m2'])
                if 'max_m2' in filters: m2_q['$lte'] = float(filters['max_m2'])
                query['m2'] = m2_q

        cursor = self.properties.find(query).sort('created_at', DESCENDING).skip(skip).limit(limit)
        return [self._serialize_doc(d) for d in cursor]

    def get_property_by_id(self, prop_id: str) -> Optional[Dict]:
        try:
            doc = self.properties.find_one({'_id': ObjectId(prop_id)})
            return self._serialize_doc(doc)
        except Exception as e:
            self.logger.error(f"get_property_by_id error: {e}")
            return None

    def update_property(self, prop_id: str, updates: Dict) -> bool:
        updates = updates.copy()
        updates['updated_at'] = datetime.now(timezone.utc)
        try:
            res = self.properties.update_one({'_id': ObjectId(prop_id)}, {'$set': updates})
            return res.modified_count > 0
        except Exception as e:
            self.logger.error(f"update_property error: {e}")
            return False

    def soft_delete_property(self, prop_id: str) -> bool:
        return self.update_property(prop_id, {'is_active': False})

    def get_property_stats(self) -> Dict:
        try:
            total = self.properties.count_documents({'is_active': True})
            sale_count = self.properties.count_documents({'is_active': True, 'listing_type': 'sale'})
            rent_count = self.properties.count_documents({'is_active': True, 'listing_type': 'rent'})
            
            pipeline = [
                {'$match': {'is_active': True}},
                {'$group': {
                    '_id': None,
                    'avg_price': {'$avg': '$price'},
                    'avg_rent': {'$avg': '$rent_price'},
                    'avg_m2': {'$avg': '$m2'}
                }}
            ]
            
            avg_result = list(self.properties.aggregate(pipeline))
            return {
                'total': total,
                'sale_count': sale_count,
                'rent_count': rent_count,
                'avg_price': int(avg_result[0]['avg_price']) if avg_result else 0,
                'avg_rent': int(avg_result[0]['avg_rent']) if avg_result else 0,
                'avg_m2': int(avg_result[0]['avg_m2']) if avg_result else 0
            }
        except Exception as e:
            self.logger.error(f"get_property_stats error: {e}")
            return {'total': 0, 'sale_count': 0, 'rent_count': 0, 'avg_price': 0, 'avg_rent': 0, 'avg_m2': 0}

    # ---------- parking ----------
    def insert_parking_analysis(self, analysis_data: Dict) -> Optional[str]:
        try:
            doc = analysis_data.copy()
            if 'property_id' in doc and isinstance(doc['property_id'], str):
                try: doc['property_id'] = ObjectId(doc['property_id'])
                except Exception: pass
            
            if 'timestamp' not in doc:
                doc['timestamp'] = datetime.now(timezone.utc)
                
            res = self.parking_analyses.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            self.logger.error(f"insert_parking_analysis error: {e}")
            return None

    def insert_district_parking_analysis(self, district_data: Dict) -> Optional[str]:
        try:
            doc = district_data.copy()
            doc['created_at'] = datetime.now(timezone.utc)
            doc['updated_at'] = datetime.now(timezone.utc)
            
            result = self.district_parking_analyses.update_one(
                {'district': doc['district']},
                {'$set': doc},
                upsert=True
            )
            
            if result.upserted_id: return str(result.upserted_id)
            else: return "updated"
        except Exception as e:
            self.logger.error(f"İlçe park analizi kaydetme hatası: {e}")
            return None

    def update_district_parking_score(self, district: str, new_score: float) -> bool:
        try:
            result = self.district_parking_analyses.update_one(
                {'district': district},
                {'$set': {
                    'summary.park_convenience_score': new_score,
                    'updated_at': datetime.now(timezone.utc)
                }}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"İlçe park skoru güncelleme hatası: {e}")
            return False

    def get_district_parking_analyses(self, district, limit=10):
        try:
            cursor = self.parking_analyses.find({'district': district}).sort([('_id', -1)]).limit(limit)
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Get district parking analyses error: {e}")
            return []

    def get_all_district_parking_analyses(self):
        try:
            pipeline = [
                {'$group': {
                    '_id': '$district',
                    'district': {'$first': '$district'},
                    'total_capacity': {'$sum': '$total_spots'},
                    'avg_empty': {'$avg': '$empty_spots'},
                    'avg_occupied': {'$avg': '$occupied_spots'},
                    'count': {'$sum': 1},
                    'last_updated': {'$max': '$timestamp'}
                }},
                {'$project': {
                    '_id': 0, 'district': 1,
                    'summary': {
                        'total_capacity': '$total_capacity',
                        'average_empty': {'$round': ['$avg_empty', 1]},
                        'average_occupied': {'$round': ['$avg_occupied', 1]},
                        'park_convenience_score': {'$multiply': [{'$divide': ['$avg_empty', '$total_capacity']}, 10]},
                        'last_updated': '$last_updated'
                    }
                }},
                {'$sort': {'district': 1}}
            ]
            return list(self.parking_analyses.aggregate(pipeline))
        except Exception as e:
            self.logger.error(f"Get all district parking analyses error: {e}")
            return []

    def get_parking_statistics(self, property_id=None, district=None):
        try:
            query = {}
            if property_id: query['property_id'] = ObjectId(property_id)
            elif district: query['district'] = district
            
            pipeline = [
                {'$match': query},
                {'$group': {
                    '_id': None,
                    'avg_empty': {'$avg': '$empty_spots'},
                    'avg_occupied': {'$avg': '$occupied_spots'},
                    'total_capacity': {'$sum': '$total_spots'},
                    'count': {'$sum': 1}
                }},
                {'$project': {
                    '_id': 0,
                    'avg_empty': {'$round': ['$avg_empty', 1]},
                    'avg_occupied': {'$round': ['$avg_occupied', 1]},
                    'total_capacity': 1, 'count': 1
                }}
            ]
            return list(self.parking_analyses.aggregate(pipeline))
        except Exception as e:
            self.logger.error(f"Get parking statistics error: {e}")
            return []

    def get_district_parking_analysis(self, district):
        try:
            analyses = list(self.parking_analyses.find({'district': district}).sort([('_id', -1)]))
            if not analyses: return None
            
            total_capacity = sum(a.get('total_spots', 0) for a in analyses)
            total_empty = sum(a.get('empty_spots', 0) for a in analyses)
            total_occupied = sum(a.get('occupied_spots', 0) for a in analyses)
            vacancy_rate = (total_empty / total_capacity) if total_capacity > 0 else 0
            park_convenience_score = round(vacancy_rate * 10, 1)
            
            images = []
            for analysis in analyses:
                images.append({
                    'original_path': f"static/parking_data/{district}/images/{analysis.get('file', '')}",
                    'result_path': f"static/results/{analysis.get('file', '')}",
                    'total_spots': analysis.get('total_spots', 0),
                    'empty_spots': analysis.get('empty_spots', 0),
                    'occupied_spots': analysis.get('occupied_spots', 0),
                    'analyzed_at': analysis.get('timestamp', '').isoformat() if hasattr(analysis.get('timestamp', ''), 'isoformat') else str(analysis.get('timestamp', ''))
                })
            
            return {
                'district': district,
                'summary': {
                    'total_capacity': total_capacity,
                    'average_empty': round(total_empty / len(analyses), 1),
                    'average_occupied': round(total_occupied / len(analyses), 1),
                    'park_convenience_score': park_convenience_score,
                    'last_updated': analyses[0].get('timestamp', '').isoformat() if analyses and hasattr(analyses[0].get('timestamp', ''), 'isoformat') else None
                },
                'images': images
            }
        except Exception as e:
            self.logger.error(f"Get district parking analysis error: {e}")
            return None

    # ---------- predictions & logs ----------
    def insert_price_prediction(self, prediction: Dict) -> Optional[str]:
        doc = prediction.copy()
        doc['created_at'] = datetime.now(timezone.utc)
        try:
            res = self.price_predictions.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            self.logger.error(f"insert_price_prediction error: {e}")
            return None

    def log_user_interaction(self, interaction: Dict) -> Optional[str]:
        doc = interaction.copy()
        doc['created_at'] = datetime.now(timezone.utc)
        try:
            res = self.user_interactions.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            self.logger.error(f"log_user_interaction error: {e}")
            return None

    def log_performance_metric(self, metric: Dict) -> Optional[str]:
        doc = metric.copy()
        doc['created_at'] = datetime.now(timezone.utc)
        try:
            res = self.metrics.insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            self.logger.error(f"log_performance_metric error: {e}")
            return None

    # ---------- simple analytics ----------
    def get_daily_stats(self, date: datetime = None) -> Dict:
        if date is None: date = datetime.now(timezone.utc).date()
        start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        try:
            props = self.properties.count_documents({'created_at': {'$gte': start, '$lt': end}})
            preds = self.price_predictions.count_documents({'created_at': {'$gte': start, '$lt': end}})
            interactions = self.user_interactions.count_documents({'created_at': {'$gte': start, '$lt': end}})
            return {'date': start.isoformat(), 'properties_created': props, 'predictions': preds, 'interactions': interactions}
        except Exception as e:
            self.logger.error(f"get_daily_stats error: {e}")
            return {}

    def get_popular_properties(self, limit: int = 10) -> List[Dict]:
        pipeline = [
            {'$group': {'_id': '$property_id', 'views': {'$sum': 1}}},
            {'$sort': {'views': -1}},
            {'$limit': limit},
            {'$lookup': {'from': 'properties', 'localField': '_id', 'foreignField': '_id', 'as': 'property'}},
            {'$unwind': {'path': '$property', 'preserveNullAndEmptyArrays': True}},
            {'$replaceRoot': {'newRoot': {'$mergeObjects': ['$property', {'views': '$views'}]}}}
        ]
        try:
            res = list(self.user_interactions.aggregate(pipeline))
            return [self._serialize_doc(r) for r in res]
        except Exception as e:
            self.logger.error(f"get_popular_properties error: {e}")
            return []