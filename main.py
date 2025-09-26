import os
import re
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    raise RuntimeError("La variable de entorno GOOGLE_API_KEY no está configurada.")

class Message(BaseModel):
    role: str
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
    candidates: Optional[List[Candidate]] = None
    history: Optional[List[Message]] = None

app = FastAPI()

def get_recommendation_from_gemini(request: RecommendationRequest):
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    chat_history = []
    if request.history:
        for msg in request.history:
            # Asegurarse de que el historial tenga el formato correcto para la API
            role = "user" if msg.role == "user" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})

    chat = model.start_chat(history=chat_history)
    
    prompt = f"Eres un asistente amigable para encontrar restaurantes llamado Tootli. Responde de forma conversacional y amigable a la consulta del usuario '{request.user_name}': '{request.user_query}'."
    
    if request.candidates and len(request.candidates) > 0:
        prompt += "\nAnaliza estos restaurantes candidatos y recomienda los que mejor encajen. Sé breve (máx 3 frases)."
        prompt += "\nAl final de tu respuesta, DEBES incluir una lista de los IDs de los restaurantes recomendados en el formato: [RECOMENDACION_IDS: id1, id2, ...]."
        prompt += "\nSi ninguno encaja, responde que no encontraste nada y devuelve [RECOMENDACION_IDS: ]."
        prompt += "\nCandidatos:"
        for c in request.candidates:
            prompt += f"\n- ID: {c.id}, Nombre: {c.name}, Descripción: {c.description}, Precio Promedio: {c.avg_price_for_two}"
    else:
        prompt += "\nNo se encontraron restaurantes para esta búsqueda. Sugiere al usuario que intente con otra consulta o más detalles. Termina tu respuesta con [RECOMENDACION_IDS: ]."

    try:
        response = chat.send_message(prompt)
        text_response = response.text
        
        # Extraer IDs de la respuesta
        ids_match = re.search(r'\[RECOMENDACION_IDS:\s*([^\]]*)\]', text_response)
        recommendation_ids = []
        if ids_match:
            ids_str = ids_match.group(1)
            if ids_str:
                recommendation_ids = [int(id.strip()) for id in ids_str.split(',') if id.strip().isdigit()]
            # Limpiar el texto de la respuesta para el usuario
            text_response = re.sub(r'\s*\[RECOMENDACION_IDS:[^\]]*\]\s*', '', text_response).strip()

        return {"responseText": text_response, "recommendation_ids": recommendation_ids}
    except Exception as e:
        print(f"Error al llamar a la API de Gemini: {e}")
        raise HTTPException(status_code=503, detail=f"El servicio de IA (Gemini) no está disponible: {str(e)}")

@app.post("/recommend")
async def recommend_dineout(request: RecommendationRequest):
    try:
        return get_recommendation_from_gemini(request)
    except ValidationError as e:
        print(f"Validation Error: {e.json()}")
        raise HTTPException(status_code=422, detail=f"Invalid request: {e.errors()}")

@app.get("/")
def read_root():
    return {"status": "Tootli AI Service is running"}