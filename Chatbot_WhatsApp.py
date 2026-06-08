import os
import re
import pickle
import unicodedata
import string
import gdown
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask_cors import CORS
import pdfplumber
from twilio.twiml.messaging_response import MessagingResponse


#SETUP INICIAL
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
load_dotenv()

#Conexiones 
try:
    client_mongo = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    client_mongo.server_info()
    db = client_mongo["chatbot_Goku_lab"]
    coleccion = db["conversaciones"]
    print("MongoDB conectado.")
except Exception as e:
    print(f"Error conectando a MongoDB: {e}")
    db = None
    coleccion = None

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]

if GROQ_KEYS:
    print(f"Groq conectado con {len(GROQ_KEYS)} key(s).")
else:
    print("No se encontraron API keys de Groq.")

#Analizador de sentimiento
analizador_sentimiento = SentimentIntensityAnalyzer()

#RAG — FALLBACK CON PDF

PDF_PATH = "gokulab_info.pdf"

def cargar_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"PDF no encontrado en: {PDF_PATH}")
        return ""
    try:
        texto = ""
        with pdfplumber.open(PDF_PATH) as pdf:
            for page in pdf.pages:
                texto += page.extract_text() + "\n"
        print(f"PDF cargado: {len(texto)} caracteres.")
        return texto
    except Exception as e:
        print(f"Error leyendo PDF: {e}")
        return ""

CONTEXTO_PDF = cargar_pdf()

#  MODELO: CARGAR O ENTRENAR
MODEL_PATH = "modelo_intents.pkl"
stop_words = set(stopwords.words("spanish"))


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join([p for p in texto.split() if p not in stop_words])


def entrenar_y_guardar():
    file_id = "1viVnkIq_QIp8jI_Ysye6Q_WVerPsvUvE"
    file_name = "intencione.xlsx"
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

    if not os.path.exists(file_name):
        print("Descargando dataset...")
        gdown.download(url, file_name, quiet=False)

    df = pd.read_excel(file_name, engine="openpyxl")
    df = df.drop(columns=["Marca temporal", "Dirección de correo electrónico"], errors="ignore")

    df_final = pd.melt(df, value_vars=df.columns, var_name="Intent", value_name="Texto")
    df_final["Intent"] = df_final["Intent"].str.strip().replace({
        "Escribe cómo preguntarías la dirección o ubicación": "Consultar_Ubicacion",
        "Escribe cómo le preguntarías a la academia cuánto cuestan los cursos": "Consultar_Costos",
        "Escribe cómo preguntarías qué horarios manejan": "Consultar_Horarios",
        "Escribe cómo preguntarías si otorgan algún certificado o diploma": "Consultar_Certificacion",
        "Escribe un saludo inicial": "Saludo",
        "Escribe cómo escribirías una despedida": "Despedida",
        "Escribe cómo preguntarías si las clases son virtuales, presenciales o mixtas": "Consultar_Modalidad",
        "Escribe cómo pedirías información sobre qué cursos tienen disponibles": "Consultar_Cursos",
        "Escribe cómo pedirías una clase demo o de prueba antes de inscribirte": "Consultar_ClaseDemo",
        "Escribe cómo preguntarías si hay edad mínima para tomar el curso": "Consultar_RequisitosEdad",
        "Escribe cómo preguntarías las formas de pago": "Consultar_FormasPago",
        "Escribe cómo preguntarías la duración de los cursos": "Consultar_Duracion",
    })
    df_final = df_final.dropna(subset=["Texto"])
    df_final["Texto"] = df_final["Texto"].apply(limpiar_texto)

    vec = TfidfVectorizer()
    X = vec.fit_transform(df_final["Texto"])
    Y = df_final["Intent"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    gs = GridSearchCV(
        SVC(probability=True),
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    gs.fit(X, Y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"modelo": gs.best_estimator_, "vectorizer": vec}, f)

    print(f"Modelo entrenado. Mejor config: {gs.best_params_}")
    return gs.best_estimator_, vec


def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        print("Modelo cargado desde disco.")
        with open(MODEL_PATH, "rb") as f:
            datos = pickle.load(f)
        return datos["modelo"], datos["vectorizer"]
    print("No se encontró modelo en disco. Entrenando...")
    return entrenar_y_guardar()


try:
    mejor_modelo, vectorizer = cargar_modelo()
except Exception as e:
    print(f"Error cargando/entrenando modelo: {e}")
    mejor_modelo, vectorizer = None, None


def predecir_intent(texto, umbral=0.5):
    if mejor_modelo is None or vectorizer is None:
        return "Desconocido", 0.0
    vector = vectorizer.transform([limpiar_texto(texto)])
    probs = mejor_modelo.predict_proba(vector)[0]
    max_prob = max(probs)
    if max_prob < umbral:
        return "Desconocido", max_prob
    return mejor_modelo.classes_[probs.argmax()], max_prob


def obtener_datos_por_intencion(intencion):
    if db is None:
        return {}

    config = db["datos_generales"].find_one({}, {"_id": 0}) or {}

    # Campos mínimos de config que siempre se usan
    config_mini = {
        "nombre_academia": config.get("nombre_academia"),
        "whatsapp":        config.get("whatsapp"),
    }

    if intencion == "Consultar_Cursos":
        cursos = list(db["cursos"].find({}, {
            "_id": 0, "nombreCurso": 1, "descripción": 1, "edad_dirigida": 1, "modalidad": 1
        }))
        return {"cursos": cursos, "config": config_mini}

    elif intencion == "Consultar_Costos":
        return {
            "costos": config.get("costos"),
            "formas_pago": config.get("formas_pago"),
            "abonos": config.get("detalle_abonos"),
            "config": config_mini
        }

    elif intencion == "Consultar_Horarios":
        horarios = list(db["horarios"].find({}, {
            "_id": 0, "nombreCurso": 1, "horarios": 1
        }))
        return {"horarios": horarios, "config": config_mini}

    elif intencion == "Consultar_Certificacion":
        return {"certificacion": config.get("certificacion"), "config": config_mini}

    elif intencion == "Consultar_ClaseDemo":
        return {"masterclass": config.get("masterclass"), "config": config_mini}

    elif intencion == "Consultar_FormasPago":
        return {
            "formas_pago": config.get("formas_pago"),
            "abonos": config.get("detalle_abonos"),
            "config": config_mini
        }

    elif intencion == "Consultar_Modalidad":
        cursos = list(db["cursos"].find({}, {
            "_id": 0, "nombreCurso": 1, "modalidad": 1
        }))
        return {"cursos": cursos, "config": config_mini}

    elif intencion == "Consultar_RequisitosEdad":
        cursos = list(db["cursos"].find({}, {
            "_id": 0, "nombreCurso": 1, "edad_dirigida": 1
        }))
        return {"cursos": cursos, "config": config_mini}

    elif intencion == "Consultar_Duracion":
        cursos = list(db["cursos"].find({}, {
            "_id": 0, "nombreCurso": 1, "duración_min_clase": 1
        }))
        return {"cursos": cursos, "config": config_mini}

    elif intencion == "Consultar_Ubicacion":
        return {
            "direccion":   config.get("direccion"),
            "referencias": config.get("referencias"),
            "maps":        config.get("google_maps"),
            "config":      config_mini
        }

    return {"config": config_mini}


#ANÁLISIS DE SENTIMIENTO

def analizar_sentimiento(texto):
    scores = analizador_sentimiento.polarity_scores(texto)
    compound = scores["compound"]
    if compound <= -0.35:
        return "negativo", compound
    elif compound >= 0.35:
        return "positivo", compound
    else:
        return "neutral", compound


#VALIDACIÓN DE ENTRADA

def validar_entrada(mensaje):
    if not mensaje or not mensaje.strip():
        return False, "empty"
    texto_limpio = re.sub(r"[^\w\s]", "", mensaje, flags=re.UNICODE).strip()
    if len(texto_limpio) < 2:
        return False, "only_symbols"
    if len(mensaje.strip()) < 2:
        return False, "too_short"
    return True, None


RESPUESTAS_INVALIDAS = {
    "empty":        "¡Hola! Parece que tu mensaje llegó vacío. ¿En qué te puedo ayudar?",
    "only_symbols": "¡Hola! No entendí bien tu mensaje. ¿Puedes escribirme tu pregunta?",
    "too_short":    "¿Puedes contarme un poco más? Con gusto te ayudo",
}


#CONSTRUCCIÓN DE PROMPTS — COMPACTOS

TONO_MAP = {
    "negativo": "El usuario está frustrado. Responde con empatía y paciencia.",
    "positivo": "El usuario está animado. Mantén esa energía.",
    "neutral":  "Responde de forma amable y profesional.",
}

INSTRUCCIONES = {
    "Saludo":                   "Saluda calurosamente, preséntate como asistente de {academia} y pregunta en qué puedes ayudar.",
    "Despedida":                "Despídete amablemente. Termina SIEMPRE con: '¡Te esperamos en Goku Lab! 🎮\nJuega, Aprende y Emprende'",
    "Desconocido":              "No entendiste la consulta. Discúlpate y pide que la reformule.",
    "Consultar_Cursos":         "Menciona los cursos disponibles con nombre y descripción breve. Sé conversacional.",
    "Consultar_Costos":         "Informa el rango de costos y formas de pago disponibles.",
    "Consultar_Horarios":       "Presenta los horarios por curso de forma clara.",
    "Consultar_Ubicacion":      "Da la dirección, referencias y link de Maps.",
    "Consultar_Modalidad":      "Explica si las clases son presenciales, online o híbridas por curso.",
    "Consultar_Certificacion":  "Explica si se otorga certificado y su validez.",
    "Consultar_ClaseDemo":      "Explica cómo funciona la clase de prueba gratuita llamada 'Master Class' ",
    "Consultar_FormasPago":     "Menciona métodos de pago y opción de abonos.",
    "Consultar_RequisitosEdad": "Explica el rango de edad por curso.",
    "Consultar_Duracion":       "Menciona la duración de cada curso.",
}

def construir_prompt(intencion, datos, config, sentimiento):
    academia = config.get("nombre_academia", "Goku Lab")
    instruccion = INSTRUCCIONES.get(intencion, f"Responde sobre: {intencion}").replace("{academia}", academia)

    return (
        f"Eres el asistente virtual de {academia}. Responde en español mexicano, natural y conciso.\n"
        f"Tono: {TONO_MAP.get(sentimiento, TONO_MAP['neutral'])}\n"
        f"Tarea: {instruccion}\n"
        f"Datos: {datos}\n"
        f"Reglas: No inventes info. Máximo 2 oraciones. Sin viñetas. Termina con una pregunta."
    )


def construir_prompt_rag(contexto_pdf, config, sentimiento):
    academia = config.get("nombre_academia", "Goku Lab")
    whatsapp = config.get("whatsapp", "")

    return (
        f"Eres el asistente virtual de {academia}. Responde en español mexicano, natural y conciso.\n"
        f"Tono: {TONO_MAP.get(sentimiento, TONO_MAP['neutral'])}\n"
        f"Usa solo esta info para responder:\n{contexto_pdf}\n"
        f"Si no tienes el dato, invita a contactar por WhatsApp: {whatsapp}.\n"
        f"Termina con una pregunta."
    )


#RESPUESTA DE EMERGENCIA

RESPUESTA_FALLBACK = (
    "En este momento tengo un problema técnico. "
    "Por favor, intenta de nuevo en un momento o escríbenos directamente por WhatsApp. 🙏"
)

def llamar_groq(messages):
    for key in GROQ_KEYS:
        try:
            cliente = Groq(api_key=key)
            respuesta = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=200,
                temperature=0.7,
                messages=messages
            )
            return respuesta.choices[0].message.content
        except Exception as e:
            print(f"Key falló: {e}. Intentando siguiente...")
            continue
    return RESPUESTA_FALLBACK

#FLASK APP

app = Flask(__name__)
CORS(app)

@app.route("/logo.png")
def logo():
    return send_file("logo.png")

@app.route("/")
def index():
    return send_file("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Body JSON requerido"}), 400

        mensaje = data.get("mensaje", "").strip()
        numero  = data.get("numero", "anonimo")

        # 1. Validación ─────────────────────────────────────
        es_valido, motivo = validar_entrada(mensaje)
        if not es_valido:
            return jsonify({
                "respuesta":   RESPUESTAS_INVALIDAS.get(motivo, "¿En qué te puedo ayudar?"),
                "intencion":   "invalido",
                "sentimiento": None,
            }), 200

        # 2. Sentimiento 
        sentimiento, score_sentimiento = analizar_sentimiento(mensaje)

        # 3. Intención 
        intencion, confianza = predecir_intent(mensaje)

        # 4. Datos mínimos según intención 
        usar_rag = intencion == "Desconocido"
        datos    = obtener_datos_por_intencion(intencion)
        config   = datos.get("config") or {}

        # 5. Historial reciente (últimos 3 intercambios) 
        historial_groq = []
        if coleccion is not None:
            historial_db = list(
                coleccion.find({"numero": numero}, {"_id": 0, "mensaje": 1, "respuesta": 1})
                .sort("timestamp", -1)
                .limit(3)
            )
            for h in reversed(historial_db):
                historial_groq.append({"role": "user",      "content": h["mensaje"]})
                historial_groq.append({"role": "assistant", "content": h["respuesta"]})

        # 6. Prompt
        if usar_rag and CONTEXTO_PDF:
            prompt_sistema = construir_prompt_rag(CONTEXTO_PDF, config, sentimiento)
        else:
            prompt_sistema = construir_prompt(intencion, datos, config, sentimiento)

        # 7. Llamada a Groq 
        respuesta = llamar_groq([
            {"role": "system", "content": prompt_sistema},
            *historial_groq,
            {"role": "user",   "content": mensaje},
        ])

        # 8. Guardar en MongoDB 
        if coleccion is not None:
            try:
                coleccion.insert_one({
                    "numero":      numero,
                    "mensaje":     mensaje,
                    "intencion":   intencion,
                    "confianza":   round(confianza, 4),
                    "sentimiento": sentimiento,
                    "score_sent":  round(score_sentimiento, 4),
                    "uso_rag":     usar_rag,
                    "respuesta":   respuesta,
                    "timestamp":   datetime.now(),
                })
            except Exception as mongo_err:
                print(f"No se pudo guardar en MongoDB: {mongo_err}")

        # 9. Respuesta 
        return jsonify({
            "intencion":   intencion,
            "confianza":   f"{confianza:.0%}",
            "sentimiento": sentimiento,
            "respuesta":   respuesta,
        })

    except Exception as e:
        print(f"Error inesperado en /chat: {e}")
        return jsonify({"respuesta": RESPUESTA_FALLBACK}), 200


@app.route("/retrain", methods=["POST"])
def retrain():
    global mejor_modelo, vectorizer
    try:
        if os.path.exists("intencione.xlsx"):
            os.remove("intencione.xlsx")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        mejor_modelo, vectorizer = entrenar_y_guardar()
        return jsonify({"status": "ok", "mensaje": "Modelo reentrenado exitosamente"}), 200
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "modelo_cargado": mejor_modelo is not None,
        "mongo_ok":       db is not None,
        "groq_ok":        len(GROQ_KEYS) > 0,
        "rag_listo":      bool(CONTEXTO_PDF),
        "timestamp":      datetime.now().isoformat(),
    }), 200


@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    try:
        mensaje = request.form.get("Body", "").strip()
        numero  = request.form.get("From", "anonimo")

        es_valido, motivo = validar_entrada(mensaje)
        if not es_valido:
            respuesta_texto = RESPUESTAS_INVALIDAS.get(motivo, "¿En qué te puedo ayudar?")
        else:
            sentimiento, score_sentimiento = analizar_sentimiento(mensaje)
            intencion, confianza = predecir_intent(mensaje)

            usar_rag = intencion == "Desconocido"
            datos    = obtener_datos_por_intencion(intencion)
            config   = datos.get("config") or {}

            historial_groq = []
            if coleccion is not None:
                historial_db = list(
                    coleccion.find({"numero": numero}, {"_id": 0, "mensaje": 1, "respuesta": 1})
                    .sort("timestamp", -1).limit(3)
                )
                for h in reversed(historial_db):
                    historial_groq.append({"role": "user",      "content": h["mensaje"]})
                    historial_groq.append({"role": "assistant", "content": h["respuesta"]})

            if usar_rag and CONTEXTO_PDF:
                prompt_sistema = construir_prompt_rag(CONTEXTO_PDF, config, sentimiento)
            else:
                prompt_sistema = construir_prompt(intencion, datos, config, sentimiento)

            respuesta_texto = llamar_groq([
                {"role": "system", "content": prompt_sistema},
                *historial_groq,
                {"role": "user",   "content": mensaje},
            ])

            if coleccion is not None:
                try:
                    coleccion.insert_one({
                        "numero":      numero,
                        "mensaje":     mensaje,
                        "intencion":   intencion,
                        "confianza":   round(confianza, 4),
                        "sentimiento": sentimiento,
                        "score_sent":  round(score_sentimiento, 4),
                        "uso_rag":     usar_rag,
                        "canal":       "whatsapp",
                        "respuesta":   respuesta_texto,
                        "timestamp":   datetime.now(),
                    })
                except Exception as e:
                    print(f"No se pudo guardar en MongoDB: {e}")

        resp = MessagingResponse()
        resp.message(respuesta_texto)
        return str(resp), 200, {"Content-Type": "text/xml"}

    except Exception as e:
        print(f"Error en /whatsapp: {e}")
        resp = MessagingResponse()
        resp.message(RESPUESTA_FALLBACK)
        return str(resp), 200, {"Content-Type": "text/xml"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Arrancando Flask en puerto {port}...")
    app.run(host="0.0.0.0", port=port)


