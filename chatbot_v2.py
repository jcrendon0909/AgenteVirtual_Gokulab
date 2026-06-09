import os
import re
import pickle
import unicodedata
import string
import gdown
import pandas as pd
import nltk
import requests
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

# ─────────────────────────────────────────────
# SETUP INICIAL
# ─────────────────────────────────────────────

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
load_dotenv()

# ─── Conexión MongoDB ───────────────────────
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

# ─── Groq keys ──────────────────────────────
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

# ─── Telegram ───────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    print("Telegram configurado.")
else:
    print("Telegram no configurado (revisa TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID).")

# ─── Analizador de sentimiento ──────────────
analizador_sentimiento = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────
# RAG — FALLBACK CON PDF
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# MODELO: CARGAR O ENTRENAR
# ─────────────────────────────────────────────

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
            "costos":      config.get("costos"),
            "formas_pago": config.get("formas_pago"),
            "abonos":      config.get("detalle_abonos"),
            "config":      config_mini,
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
            "abonos":      config.get("detalle_abonos"),
            "config":      config_mini,
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
            "config":      config_mini,
        }

    return {"config": config_mini}


ETIQUETAS_INTENCION = {
    "Consultar_Costos":     "Consulta de precios",
    "Consultar_ClaseDemo":  "Clase demo / Master Class",
}

def notificar_marco(numero_usuario, intencion, mensaje_original):
    notificar_marco_con_contexto(numero_usuario, intencion, mensaje_original, "")

def notificar_marco_con_contexto(numero_usuario, intencion, mensaje_original, contexto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram no configurado, notificación omitida.")
        return

    tema = ETIQUETAS_INTENCION.get(intencion, intencion)

    texto = (
        f"🔔 *Nuevo lead — Gōku Lab*\n\n"
        f"Tema: {tema}\n"
        f"contacto: `{numero_usuario}`\n"
        f"¿Qué consultó?: _{mensaje_original}_"
    )

    if contexto:
        texto += f"\n\n📋 *Conversación previa:*\n{contexto}"

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       texto,
                "parse_mode": "Markdown",
            },
            timeout=5,
        )
        if resp.ok:
            print("Notificación Telegram enviada.")
        else:
            print(f"Error Telegram: {resp.text}")
    except Exception as e:
        print(f"Error enviando notificación Telegram: {e}")


INTENCIONES_REQUIEREN_HUMANO = {
    "Consultar_Costos",
    "Consultar_ClaseDemo",
}

# ─────────────────────────────────────────────
# ANÁLISIS DE SENTIMIENTO
# ─────────────────────────────────────────────

def analizar_sentimiento(texto):
    scores = analizador_sentimiento.polarity_scores(texto)
    compound = scores["compound"]
    if compound <= -0.35:
        return "negativo", compound
    elif compound >= 0.35:
        return "positivo", compound
    else:
        return "neutral", compound


# ─────────────────────────────────────────────
# VALIDACIÓN DE ENTRADA
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# CONSTRUCCIÓN DE PROMPTS
# ─────────────────────────────────────────────

TONO_MAP = {
    "negativo": "El usuario está frustrado. Responde con empatía y paciencia.",
    "positivo": "El usuario está animado. Mantén esa energía.",
    "neutral":  "Responde de forma amable y profesional.",
}

INSTRUCCIONES = {
    "Saludo": "Saluda calurosamente, preséntate como asistente de {academia} y pregunta en qué puedes ayudar.",
    "Despedida": (
        "El usuario se está despidiendo. "
        "Despídete de forma breve y amable. "
        "NO hagas preguntas. NO menciones teléfono, WhatsApp ni correos. "
        "Tu respuesta DEBE terminar EXACTAMENTE con esta frase, sin cambiarla: "
        "'¡Te esperamos en Gōku Lab! 🎮\nJuega, Aprende y Emprende'"
    ),
    "Desconocido":             "No entendiste la consulta. Discúlpate y pide que la reformule.",
    "Consultar_Cursos":        "Menciona los cursos disponibles con nombre y descripción breve (máximo dos líneas). Sé conversacional.",
    "Consultar_Costos": "Da el rango de costos en UNA sola oración muy breve. NO inventes precios exactos. NO menciones WhatsApp ni correos.",
    "Consultar_Horarios":      "Presenta los horarios por curso de forma clara.",
    "Consultar_Ubicacion":     "Da la dirección, referencias y link de Maps.",
    "Consultar_Modalidad":     "Explica si las clases son presenciales, online o híbridas por curso.",
    "Consultar_Certificacion": "Explica si se otorga certificado y su validez.",
    "Consultar_ClaseDemo": (
        "Explica que existe una Master Class gratuita para conocer la metodología. "
        "NO menciones correos, enlaces, formularios ni WhatsApp. "
        "NO inventes fechas ni horarios fijos."
    ),
    "Consultar_FormasPago":    "Menciona métodos de pago y opción de abonos.",
    "Consultar_RequisitosEdad":"Explica el rango de edad por curso.",
    "Consultar_Duracion": (
        "Explica que cada clase tiene una duración de 90 minutos y se imparte una vez por semana. "
        "Menciona que el cliente puede elegir inscribir a su hijo en más de una sesión semanal. "
        "NO inventes horarios ni días específicos. "
        "Invita a preguntar sobre horarios disponibles."
    ),
}

def construir_prompt(intencion, datos, config, sentimiento):
    academia = config.get("nombre_academia", "GōkuLab")
    instruccion = INSTRUCCIONES.get(intencion, f"Responde sobre: {intencion}").replace("{academia}", academia)

    return (
        f"Eres el asistente virtual de {academia}. Responde en español mexicano, natural y conciso.\n"
        f"Tono: {TONO_MAP.get(sentimiento, TONO_MAP['neutral'])}\n"
        f"Tarea: {instruccion}\n"
        f"Datos: {datos}\n"
        f"Reglas: No inventes info. MÁXIMO 2 oraciones. Sin viñetas. "
        f"Si el usuario hace más de una pregunta y tienes los datos, responde ambas. "
        f"Si el usuario se despide NO hagas preguntas. "
        f"Termina con una pregunta SOLO si NO es despedida."
    )


def construir_prompt_rag(contexto_pdf, config, sentimiento):
    academia = config.get("nombre_academia", "GōkuLab")

    return (
        f"Eres el asistente virtual de {academia}. Responde en español mexicano, natural y conciso.\n"
        f"Tono: {TONO_MAP.get(sentimiento, TONO_MAP['neutral'])}\n"
        f"Usa solo esta info para responder:\n{contexto_pdf}\n"
        f"Si el usuario se despide NO hagas preguntas. "
        f"Termina con una pregunta SOLO si NO es despedida."
    )


# ─────────────────────────────────────────────
# RESPUESTA DE EMERGENCIA
# ─────────────────────────────────────────────

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
                max_tokens=120,
                temperature=0.7,
                messages=messages,
            )
            return respuesta.choices[0].message.content
        except Exception as e:
            print(f"Key falló: {e}. Intentando siguiente...")
            continue
    return RESPUESTA_FALLBACK

# ================== LÓGICA CENTRAL DEL CHATBOT ==================
def procesar_mensaje(numero: str, mensaje: str) -> dict:
    """
    Retorna un diccionario con:
    - respuesta: str
    - intencion: str
    - confianza: str
    - sentimiento: str
    """
    # 1. Validación
    es_valido, motivo = validar_entrada(mensaje)
    if not es_valido:
        return {
            "respuesta":   RESPUESTAS_INVALIDAS.get(motivo, "¿En qué te puedo ayudar?"),
            "intencion":   "invalido",
            "confianza":   "0%",
            "sentimiento": None,
        }

    # 2. ¿Estábamos esperando el número del usuario?
    esperando_numero    = False
    intencion_pendiente = None
    mensaje_original    = None

    if db is not None:
        estado = db["estados"].find_one({"numero": numero})
        if estado and estado.get("esperando_numero"):
            esperando_numero    = True
            intencion_pendiente = estado.get("intencion_pendiente")
            mensaje_original    = estado.get("mensaje_original", "")

    if esperando_numero:
        solo_numeros = re.sub(r"[\s\-\(\)\+]", "", mensaje)
        es_numero = solo_numeros.isdigit() and len(solo_numeros) >= 8

        if not es_numero:
            return {
                "intencion":   "esperando_numero",
                "confianza":   "100%",
                "sentimiento": "neutral",
                "respuesta":   "Para conectarte con nuestro equipo necesito tu número de WhatsApp. ¿Me lo compartes? 😊",
            }

        numero_dado = mensaje

        contexto_conversacion = ""
        if coleccion is not None:
            historial_lead = list(
                coleccion.find({"numero": numero}, {"_id": 0, "mensaje": 1, "respuesta": 1})
                .sort("timestamp", -1)
                .limit(4)
            )
            if historial_lead:
                lineas = []
                for h in reversed(historial_lead):
                    lineas.append(f"Usuario: {h['mensaje']}")
                    lineas.append(f"Bot: {h['respuesta']}")
                contexto_conversacion = "\n".join(lineas)

        #notificar_marco_con_contexto(numero_dado, intencion_pendiente, mensaje_original, contexto_conversacion)

        db["estados"].delete_one({"numero": numero})

        if coleccion is not None:
            try:
                coleccion.insert_one({
                    "numero":      numero_dado,
                    "mensaje":     f"[número capturado] {numero_dado}",
                    "intencion":   "captura_numero",
                    "confianza":   1.0,
                    "sentimiento": "neutral",
                    "score_sent":  0.0,
                    "uso_rag":     False,
                    "respuesta":   "Número enviado al equipo.",
                    "timestamp":   datetime.now(),
                })
            except Exception as mongo_err:
                print(f"No se pudo guardar en MongoDB: {mongo_err}")

        return {
            "intencion":   "captura_numero",
            "confianza":   "100%",
            "sentimiento": "neutral",
            "respuesta": (
                "¡Listo! Nuestro equipo se pondrá en contacto contigo muy pronto. "
                "¿Hay algo más en lo que pueda ayudarte?"
            ),
        }

    # 3. Sentimiento
    sentimiento, score_sentimiento = analizar_sentimiento(mensaje)

    # 4. Intención
    intencion, confianza = predecir_intent(mensaje)

    # 5. ¿Esta intención requiere atención humana?
    if intencion in INTENCIONES_REQUIEREN_HUMANO:
        ya_dio_numero = False
        if coleccion is not None:
            captura_previa = coleccion.find_one({
                "numero":    numero,
                "intencion": "captura_numero",
            })
            if captura_previa:
                ya_dio_numero = True

        if ya_dio_numero:
            datos  = obtener_datos_por_intencion(intencion)
            config = datos.get("config") or {}
            respuesta_directa = llamar_groq([
                {"role": "system", "content": construir_prompt(intencion, datos, config, sentimiento)},
                {"role": "user",   "content": mensaje},
            ])
            return {
                "intencion":   intencion,
                "confianza":   f"{confianza:.0%}",
                "sentimiento": sentimiento,
                "respuesta":   respuesta_directa,
            }

        if db is not None:
            db["estados"].replace_one(
                {"numero": numero},
                {
                    "numero":              numero,
                    "esperando_numero":    True,
                    "intencion_pendiente": intencion,
                    "mensaje_original":    mensaje,
                },
                upsert=True,
            )

        datos  = obtener_datos_por_intencion(intencion)
        config = datos.get("config") or {}

        respuesta_parcial = llamar_groq([
            {"role": "system", "content": construir_prompt(intencion, datos, config, sentimiento)},
            {"role": "user",   "content": mensaje},
        ])

        return {
            "intencion":   intencion,
            "confianza":   f"{confianza:.0%}",
            "sentimiento": sentimiento,
            "respuesta": (
                f"{respuesta_parcial}\n\n"
                "¿Me compartes tu número de WhatsApp para darte info personalizada?"
            ),
        }

    # 6. Flujo normal (sin requerir humano)
    usar_rag = intencion == "Desconocido"
    datos    = obtener_datos_por_intencion(intencion)
    config   = datos.get("config") or {}

    # Historial reciente
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

    if usar_rag and CONTEXTO_PDF:
        prompt_sistema = construir_prompt_rag(CONTEXTO_PDF, config, sentimiento)
    else:
        prompt_sistema = construir_prompt(intencion, datos, config, sentimiento)

    respuesta = llamar_groq([
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
                "respuesta":   respuesta,
                "timestamp":   datetime.now(),
            })
        except Exception as mongo_err:
            print(f"No se pudo guardar en MongoDB: {mongo_err}")

    return {
        "intencion":   intencion,
        "confianza":   f"{confianza:.0%}",
        "sentimiento": sentimiento,
        "respuesta":   respuesta,
    }

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


# @app.route("/logo.png")
#def logo():
 #   return send_file("logo.png")


@app.route("/")
def index():
   return jsonify({"status": "ok", "message": "Chatbot API is running"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Body JSON requerido"}), 400

        mensaje = data.get("mensaje", "").strip()
        numero  = data.get("numero", "anonimo")

        resultado = procesar_mensaje(numero, mensaje)
        return jsonify(resultado), 200

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
        "status":          "ok",
        "modelo_cargado":  mejor_modelo is not None,
        "mongo_ok":        db is not None,
        "groq_ok":         len(GROQ_KEYS) > 0,
        "rag_listo":       bool(CONTEXTO_PDF),
        "telegram_ok":     bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "timestamp":       datetime.now().isoformat(),
    }), 200

# ================== WEBHOOKS PARA MULTICANAL ==================
# Dependencias: python-telegram-bot, pywa, requests (ya instalado)

# ---------- TELEGRAM ----------
from telegram import Update
import telegram

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Token del bot (mismo que TELEGRAM_TOKEN)
if TELEGRAM_BOT_TOKEN:
    telegram_bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    print("Telegram bot configurado para webhook.")
else:
    telegram_bot = None
    print("Telegram bot no configurado (falta TELEGRAM_BOT_TOKEN).")

@app.route("/webhook/telegram", methods=["POST"])
def telegram_webhook():
    if telegram_bot is None:
        return "Telegram bot not configured", 500
    try:
        update = Update.de_json(request.get_json(force=True), telegram_bot)
        if update.message and update.message.text:
            chat_id = update.message.chat.id
            user_text = update.message.text
            resultado = procesar_mensaje(str(chat_id), user_text)
            telegram_bot.send_message(chat_id=chat_id, text=resultado["respuesta"])
        return "OK", 200
    except Exception as e:
        print(f"Error en webhook Telegram: {e}")
        return "Error", 500

# ---------- WHATSAPP BUSINESS ----------
from pywa import WhatsApp

WA_PHONE_ID = os.getenv("WA_PHONE_ID")
WA_ACCESS_TOKEN = os.getenv("WA_ACCESS_TOKEN")
WA_APP_ID = os.getenv("WA_APP_ID")
WA_APP_SECRET = os.getenv("WA_APP_SECRET")
WA_VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN", "gokulab_wa_verify")

if all([WA_PHONE_ID, WA_ACCESS_TOKEN, WA_APP_ID, WA_APP_SECRET]):
    wa_client = WhatsApp(
        phone_id=WA_PHONE_ID,
        token=WA_ACCESS_TOKEN,
        app_id=WA_APP_ID,
        app_secret=WA_APP_SECRET,
    )
    print("WhatsApp Business configurado.")
else:
    wa_client = None
    print("Faltan variables para WhatsApp Business.")

@app.route("/webhook/whatsapp", methods=["GET", "POST"])
def whatsapp_webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == WA_VERIFY_TOKEN:
            return challenge, 200
        return "Verification failed", 403

    if wa_client is None:
        return "WhatsApp not configured", 500

    try:
        data = request.get_json()
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        if "messages" in value:
            message = value["messages"][0]
            from_number = message["from"]
            text = message["text"]["body"]
            resultado = procesar_mensaje(from_number, text)
            wa_client.send_text(to=from_number, text=resultado["respuesta"])
        return "OK", 200
    except Exception as e:
        print(f"Error en webhook WhatsApp: {e}")
        return "Error", 500

# ---------- MESSENGER / INSTAGRAM ----------
PAGE_ACCESS_TOKEN = os.getenv("META_PAGE_ACCESS_TOKEN")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "gokulab_meta_verify")

@app.route("/webhook/meta", methods=["GET", "POST"])
def meta_webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == META_VERIFY_TOKEN:
            return challenge, 200
        return "Verification failed", 403

    if not PAGE_ACCESS_TOKEN:
        return "Meta not configured", 500

    try:
        data = request.get_json()
        for entry in data.get("entry", []):
            for messaging in entry.get("messaging", []):
                sender_id = messaging["sender"]["id"]
                if "message" in messaging and "text" in messaging["message"]:
                    user_text = messaging["message"]["text"]
                    resultado = procesar_mensaje(sender_id, user_text)
                    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
                    payload = {
                        "recipient": {"id": sender_id},
                        "message": {"text": resultado["respuesta"]}
                    }
                    requests.post(url, json=payload)
        return "OK", 200
    except Exception as e:
        print(f"Error en webhook Meta: {e}")
        return "Error", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Arrancando Flask en puerto {port}...")
    app.run(host="0.0.0.0", port=port)