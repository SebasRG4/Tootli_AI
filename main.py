import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# --- CONFIGURACIÓN ---
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    raise RuntimeError("La variable de entorno GOOGLE_API_KEY no está configurada.")

# --- MODELOS DE DATOS ---
class Message(BaseModel):
    role: str  # 'user' o 'model'
    content: str

class Candidate(BaseModel):
    id: int
    name: str
    address: str
    avg_price_for_two: float
    description: str

class RecommendationRequest(BaseModel):
    user_query: str
    user_name: str
    candidates: Optional[List[Candidate]] = None  # Opcional para turnos posteriores
    history: Optional[List[Message]] = None  # Historial de mensajes previos

# --- INICIALIZACIÓN DE FASTAPI ---
app = FastAPI()

# --- LÓGICA DE LA IA CON GOOGLE GEMINI ---
def get_recommendation_from_gemini(request: RecommendationRequest):
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Preparar el historial para el chat (si no hay, inicia vacío)
    chat_history = []
    if request.history:
        for msg in request.history:
            chat_history.append({"role": msg.role, "parts": [msg.content]})

    # Iniciar el chat con el historial
    chat = model.start_chat(history=chat_history)

    # Crear el prompt inicial/conversacional
    prompt = f"Responde de forma conversacional y amigable a la consulta del usuario '{request.user_name}': '{request.user_query}'."
    if request.candidates:  # Solo incluye candidatos en la primera llamada
        prompt += "\nAnaliza estos restaurantes candidatos y recomienda el mejor (o ninguno si no encaja). Sé breve (máx 3 frases)."
        for c in request.candidates:
            prompt += f"\n- ID: {c.id}, Nombre: {c.name}, Descripción: {c.description}, Precio Promedio: {c.avg_price_for_two}"
    else:
        prompt += "\nContinúa la conversación basada en el historial, recomendando o aclarando si es necesario."

    try:
        # Enviar el prompt al chat y obtener la respuesta
        response = chat.send_message(prompt)
        return {"responseText": response.text}

    except Exception as e:
        print(f"Error al llamar a la API de Gemini: {e}")
        raise HTTPException(status_code=503, detail="El servicio de IA (Gemini) no está disponible.")

# --- ENDPOINT DE LA API ---
@app.post("/recommend")
async def recommend_dineout(request: RecommendationRequest):
    return get_recommendation_from_gemini(request)

@app.get("/")
def read_root():
    return {"status": "Tootli AI Service is running"}