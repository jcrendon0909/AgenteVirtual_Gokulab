# test_local.py
import os
from dotenv import load_dotenv
import traceback

# Cargar variables de entorno
load_dotenv()

# Importar la función principal (ajusta el nombre según tu archivo)
# Si tu archivo principal es chatbot_v2.py:
from chatbot_v2 import procesar_mensaje  # O el nombre que tenga tu función

def test_bot():
    print("🧪 Probando el bot localmente...")
    print("="*50)
    
    # Prueba 1: Mensaje simple
    try:
        print("📝 Prueba 1: 'Hola, ¿qué cursos ofrecen?'")
        respuesta = procesar_mensaje("Hola, ¿qué cursos ofrecen?", "test_user_123")
        print(f"✅ Respuesta: {respuesta}")
        print("="*50)
    except Exception as e:
        print("❌ ERROR en Prueba 1:")
        print(traceback.format_exc())
    
    # Prueba 2: Mensaje específico
    try:
        print("\n📝 Prueba 2: 'Quiero información sobre robótica'")
        respuesta = procesar_mensaje("Quiero información sobre robótica", "test_user_123")
        print(f"✅ Respuesta: {respuesta}")
        print("="*50)
    except Exception as e:
        print("❌ ERROR en Prueba 2:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_bot()