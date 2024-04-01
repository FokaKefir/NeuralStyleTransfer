import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('utils/google-services.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
gens_ref = db.collection('gens')

def update_gen_document(doc_id: str, image_url: str):
    doc_ref = gens_ref.document(doc_id)

    doc_ref.update({"outputImage": image_url})