import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("mongodb+srv://gokulabadmin:admin1234@clustergokulab.klpb0nj.mongodb.net/gestion_academica?appName=Clustergokulab")
if not uri:
    print("MONGO_URI no encontrada en .env")
    exit(1)

client = MongoClient(uri)
db = client["chatbot_Goku_lab"]
coleccion = db["datos_generales"]

coleccion.update_one(
    {},
    {"$set": {"nombre_academia": "Gōku Lab"}},
    upsert=True
)
print("Documento 'datos_generales' creado/actualizado correctamente.")