# 🗺️ Roadmap — Agente GōkuLab multicanal "de verdad"

**Principio rector:** cada fase es independiente y reversible. Telegram (lo único que hoy funciona en producción) **no se toca** hasta la Fase 5. Si algo en una fase falla, Telegram sigue funcionando porque vive en su propio webhook.

---

## ✅ FASE -1 (ya hecha)
Detección de hasta 3 intenciones por mensaje + prompts mejorados de Valeria — esto ya está integrado en el `chatbot_v2.py` que tienes corriendo. No requiere acción.

---

## 🟢 FASE 1 — Sacar el PDF, meterlo a MongoDB

**Por qué primero:** es la fase de menor riesgo, no agrega canales nuevos, y mejora la calidad de TODAS las respuestas que hoy usan RAG (21% de los mensajes).

### Paso 1.1 — Importar la nueva colección
Te generé `conocimiento.json` con las 21 secciones del PDF, ya troceadas por tema (FAQs, redes sociales, directores, política de inclusión, etc.), cada una con `palabras_clave` para el futuro.

En **MongoDB Compass**:
1. Conéctate a tu cluster de producción
2. En `chatbot_Goku_lab`, crea la colección `conocimiento`
3. **Add Data → Import JSON** → selecciona `conocimiento.json`

### Paso 1.2 — Un solo punto a revisar antes de publicar
El documento `inclusion_necesidades_especiales` (política sobre alumnos con autismo) lo marqué con `REVISAR_CON_DIRECCION` — es un tema sensible que hoy está como una sola línea suelta en el PDF. Antes de que el bot lo use, revísalo con Claudia y redáctenlo con el detalle que ambos consideren apropiado. Mientras no lo editen, el bot seguirá funcionando normal (solo no dará mucho detalle si alguien pregunta sobre esto).

### Paso 1.3 — Cambiar el código (3 líneas)

En `chatbot_v2.py`, reemplaza:

```python
PDF_PATH = "gokulab_info.pdf"

def cargar_pdf():
    if not os.path.exists(PDF_PATH):
        ...
    # ... lee el PDF con pdfplumber

CONTEXTO_PDF = cargar_pdf()
```

por:

```python
def cargar_conocimiento():
    if db is None:
        return ""
    docs = db["conocimiento"].find({}, {"_id": 0, "contenido": 1})
    return "\n\n".join(d["contenido"] for d in docs)

CONTEXTO_PDF = cargar_conocimiento()
```

**No cambies el nombre `CONTEXTO_PDF`** por ahora — así `construir_prompt_rag(CONTEXTO_PDF, ...)` sigue funcionando sin tocar nada más. (Lo renombras más adelante si quieres, con calma.)

### Paso 1.4 — Probar y limpiar
```bash
python chatbot_v2.py
```
Prueba en Telegram preguntas que antes dependían del PDF: "¿tienen estacionamiento?", "¿qué redes sociales tienen?". Si responde bien:

```bash
git rm gokulab_info.pdf
# quita pdfplumber de requirements.txt si no se usa en otro lado
git add . && git commit -m "feat: mover base de conocimiento del PDF a MongoDB" && git push
```

✅ **Telegram no se ve afectado** — sigue usando el mismo `procesar_mensaje()`.

---

## 🟢 FASE 2 — Widget web en gokulab.mx

### Diseño de la mejora
En vez de un iframe (lento, difícil de estilizar), un **componente flotante en React** dentro de `GokuLab_WebSite` que llama directo a tu API `/chat` — ya tienes CORS configurado.

### Paso 2.1 — Identidad de sesión real
Hoy el widget mandaría `numero: "anonimo"` para todos — eso mezclaría el historial de todos los visitantes. Arréglalo generando un ID único por visitante:

```tsx
// src/lib/chatSession.ts
export function getChatSessionId(): string {
  let id = localStorage.getItem("gokulab_chat_id");
  if (!id) {
    id = "web_" + crypto.randomUUID();
    localStorage.setItem("gokulab_chat_id", id);
  }
  return id;
}
```

### Paso 2.2 — Componente flotante
Burbuja fija en la esquina inferior derecha, con `fetch` a `https://agentevirtual-gokulab.onrender.com/chat`. Cuando estés listo para esta fase, te genero el componente completo (`GokuChatWidget.tsx`) con shadcn/ui para que combine con el resto del sitio.

### Paso 2.3 — CORS
Verifica que `CORS(app)` en `chatbot_v2.py` permita el dominio de producción. Si quieres restringirlo (recomendado antes de ir a producción):

```python
CORS(app, origins=["https://gokulab.mx", "https://www.gokulab.mx"])
```

✅ **Telegram no se ve afectado** — es un canal nuevo, aislado.

---

## 🟡 FASE 3 — WhatsApp Business (completar lo ya empezado)

Tu código (`/webhook/whatsapp` con `pywa`) ya está. Falta la configuración en Meta:

### Paso 3.1 — Meta Business Manager
1. business.facebook.com → crear/usar tu Business Manager
2. Meta for Developers → crear una App tipo "Business"
3. Agregar el producto WhatsApp

### Paso 3.2 — Token permanente
El token de prueba de WhatsApp expira en 24h. Para producción necesitas un token permanente del sistema (System User):
1. Business Settings → System Users → crear uno
2. Asignar el activo de WhatsApp
3. Generar token con permiso whatsapp_business_messaging

### Paso 3.3 — Configurar el webhook en Meta
- URL: https://agentevirtual-gokulab.onrender.com/webhook/whatsapp
- Verify token: el mismo valor de WA_VERIFY_TOKEN en Render
- Suscribirse al campo messages

### Paso 3.4 — Variables en Render
Confirma que estén: WA_PHONE_ID, WA_ACCESS_TOKEN (el permanente), WA_APP_ID, WA_APP_SECRET, WA_VERIFY_TOKEN

### Paso 3.5 — Probar
Manda un mensaje al número de WhatsApp Business desde tu celular personal. Revisa logs de Render.

✅ **Telegram no se ve afectado** — webhook independiente.

---

## 🟡 FASE 4 — Messenger + Instagram

**La buena noticia:** usan la MISMA App de Meta que configuraste en la Fase 3, y tu código ya tiene `/webhook/meta` listo para ambos.

### Paso 4.1 — Agregar productos a la App
En la misma App de Meta (Fase 3) → agregar productos Messenger e Instagram

### Paso 4.2 — Conectar cuentas
- Vincula la página de Facebook de GōkuLab
- Vincula la cuenta de Instagram profesional (debe estar conectada a esa página de Facebook)

### Paso 4.3 — Webhook
- Misma URL: https://agentevirtual-gokulab.onrender.com/webhook/meta
- Suscríbete a messages (Messenger) y messages (Instagram)
- Verify token: el de META_VERIFY_TOKEN

### Paso 4.4 — Variable en Render
Confirma META_PAGE_ACCESS_TOKEN (token de la página, con permisos pages_messaging + instagram_manage_messages)

✅ **Telegram, WhatsApp y el widget no se ven afectados.**

---

## 🟡 FASE 5 — Colección leads (CRM básico)

Hoy los leads (alguien que preguntó costos/demo y dejó su número) quedan mezclados dentro de conversaciones con intencion: "captura_numero". Para dar seguimiento real, separémoslo:

```python
# Dentro de procesar_mensaje, donde hoy se hace el insert de "captura_numero"
db["leads"].insert_one({
    "numero":            numero_dado,
    "canal":             canal,          # "telegram" | "whatsapp" | "web" | "messenger" | "instagram"
    "intencion_origen":  intencion_pendiente,
    "mensaje_original":  mensaje_original,
    "estado":            "nuevo",        # nuevo | contactado | convertido | descartado
    "timestamp":         datetime.now(),
})
```

Esto requiere agregar un parámetro `canal` a `procesar_mensaje(numero, mensaje, canal)` — pequeño cambio, cada webhook ya sabe de qué canal viene.

**Beneficio a futuro:** esta colección es el puente natural hacia gestion_academica — un lead convertido puede convertirse en inscripción con un solo paso de import.

✅ **No afecta canales existentes** — solo agrega un registro adicional.

---

## 🔵 FASE 6 — TikTok (al final, y con expectativas realistas)

Investigué el estado actual de la API de TikTok para negocios:

- Requiere cuenta TikTok Business (no personal)
- El usuario debe escribir primero — el bot no puede iniciar conversación
- El acceso a la Business Messaging API real (webhooks de DM) pasa por un proceso de aprobación de TikTok, más lento que Meta
- No está disponible en UE/Reino Unido/Suiza (no afecta a México)

**Recomendación:** mientras la API se aprueba, usa las respuestas automáticas nativas de TikTok (configurables desde la app de TikTok Business: responder por palabra clave a comentarios y DMs) como solución puente. Cuando tengas acceso a la API real, conectamos un /webhook/tiktok igual que los demás.

---

## 📋 Tabla resumen

| Fase | Qué logras | Riesgo para lo existente | Esfuerzo |
|---|---|---|---|
| 1 | PDF fuera, conocimiento editable en Mongo | Ninguno | Bajo |
| 2 | Chat en gokulab.mx | Ninguno (canal nuevo) | Medio |
| 3 | WhatsApp funcionando | Ninguno (webhook propio) | Medio |
| 4 | Messenger + Instagram | Ninguno (mismo setup que 3) | Bajo (incremental) |
| 5 | CRM básico de leads | Ninguno (solo agrega datos) | Bajo |
| 6 | TikTok | Ninguno | Alto / depende de TikTok |

---

## 🚦 Por dónde empezamos

Mi sugerencia: Fase 1 hoy mismo (es la de menor riesgo y mejora todo lo demás), y en paralelo das de alta la App de Meta (Fase 3, paso 3.1) porque la aprobación de Meta puede tardar un par de días — así cuando lleguemos al código de WhatsApp/Messenger/Instagram, la parte administrativa ya está lista.

¿Arrancamos con la Fase 1?
