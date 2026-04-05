# test_db.py
from database import MongoDatabaseManager

def test_insert_and_fetch():
    db = MongoDatabaseManager()
    db.create_collections_and_indexes()

    # örnek property
    prop_data = {
        'price': 8500000,
        'price_per_m2': 120000,
        'm2': 70,
        'rooms': 3,
        'location_tier': 2,
        'district': 'Kadıköy',
        'neighborhood': 'Moda',
        'furnished': True,
        'air_conditioning': True,
        'has_balcony': True,
        'has_elevator': True,
        'has_parking': True
    }

    prop_id = db.insert_property(prop_data)
    print(f"Inserted property ID: {prop_id}")

    prop = db.get_property_by_id(prop_id)
    print("Fetched property:")
    print(prop)

if __name__ == "__main__":
    test_insert_and_fetch()
