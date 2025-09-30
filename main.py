import os
import re
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any

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
    tags: Optional[List[str]] = Field(default_factory=list)
    discount_info: Optional[str] = None
    rating: Optional[float] = None
    serves_alcohol: Optional[bool] = None
    featured: Optional[bool] = None
    delivery_time: Optional[str] = None
    tipo_cocina: Optional[str] = None

    class Config:
        extra = "ignore"

class RecommendationRequest(BaseModel):
    user_query: str
    user_name: str
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    candidates: Optional[List[Candidate]] = None
    history: Optional[List[Message]] = None
    previous_candidate_ids: Optional[List[int]] = None

    class Config:
        extra = "ignore"

app = FastAPI()

def get_recommendation_from_gemini(request: RecommendationRequest):
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    chat_history = []
    if request.history:
        for msg in request.history:
            role = "user" if msg.role == "user" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})

    chat = model.start_chat(history=chat_history)

    # PROMPT MEJORADO CON TAGS Y REFINAMIENTO
    prompt = f"""¡Hola! Eres Toot, un asistente amigable que ayuda a encontrar restaurantes.

Consulta: "{request.user_query}"
Filtros aplicados: {request.filters if request.filters else 'Ninguno'}

REGLAS OBLIGATORIAS:
1) SOLO puedes recomendar restaurantes que estén en la lista "RESTAURANTES DISPONIBLES" que te proveo abajo.
2) Si el request incluye "previous_candidate_ids", considera que el usuario está refinando una búsqueda previa. En ese caso debes RECOMENDAR SOLO aquellos restaurantes que:
   a) estén en la lista de candidatos proporcionada y
   b) además estén dentro de previous_candidate_ids (es decir, intersecta la nueva elección con los previos).
   Si no existe intersección, devuelve una lista vacía de IDs y un breve mensaje de fallback (ej.: "No hay restaurantes que cumplan todas las condiciones; ¿quieres ver opciones que cumplan la última condición?").
3) Al final de tu respuesta devuelve exactamente el token: [RECOMENDACION_IDS: id1, id2, ...] (si no hay ids, escribe [RECOMENDACION_IDS:]).
4) En el texto explica brevemente (1-2 frases por restaurante) por qué lo recomiendas, pero NO incluyas IDs dentro del texto.
5) NO uses negritas ni formateo con IDs.

"""

    if request.candidates and len(request.candidates) > 0:
        prompt += f"\n\nRESTAURANTES DISPONIBLES ({len(request.candidates)} encontrados):"
        for c in request.candidates:
            features = []
            if getattr(c, 'featured', None):
                features.append("⭐ Destacado")
            if getattr(c, 'serves_alcohol', None):
                features.append("🍷 Sirve alcohol")
            if getattr(c, 'rating', None) and c.rating >= 4.0:
                features.append(f"🌟 Rating: {c.rating}")
            if getattr(c, 'delivery_time', None):
                features.append(f"⏱️ {c.delivery_time}")

            features_str = " | ".join(features) if features else "Estándar"
            tags = getattr(c, 'tags', [])
            tags_str = " | ".join(tags) if tags else "Sin tags específicos"
            discount_info = getattr(c, 'discount_info', None)
            tipo_cocina = getattr(c, 'tipo_cocina', 'Cocina variada')

            prompt += f"""
- ID: {c.id} | {c.name}
  💰 ${c.avg_price_for_two} para dos | {tipo_cocina}
  🏷️ TAGS: {tags_str}
  ✨ {features_str}{f" | 🎯 {discount_info}" if discount_info else ""}
  📍 {c.address}"""

        prompt += f"\n\nANÁLISIS: Basándote en los TAGS y características, recomienda los que mejor coincidan con '{request.user_query}'."
    else:
        prompt += "\n\nNo se encontraron restaurantes para esta búsqueda. Sugiere al usuario que intente con otros términos o menos filtros. Termina tu respuesta con [RECOMENDACION_IDS: ]."

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

        # DEFENSA: asegurar que las ids devueltas pertenecen a los candidates
        valid_candidate_ids = [c.id for c in request.candidates] if request.candidates else []
        recommendation_ids = [rid for rid in recommendation_ids if rid in valid_candidate_ids]

        # Si previous_candidate_ids fue enviada, aplicar intersección estricta
        if request.previous_candidate_ids:
            previous_ids = [int(x) for x in request.previous_candidate_ids]
            intersection = [rid for rid in recommendation_ids if rid in previous_ids]
            if intersection:
                recommendation_ids = intersection
            else:
                # No hay intersección: devolver lista vacía y mensaje de fallback
                recommendation_ids = []
                text_response = "No hay restaurantes que cumplan todas las condiciones. ¿Quieres ver resultados que solo cumplan la última condición?"

        return {"responseText": text_response, "recommendation_ids": recommendation_ids}
    except Exception as e:
        print(f"Error al llamar a la API de Gemini: {e}")
        raise HTTPException(status_code=503, detail=f"El servicio de IA (Gemini) no está disponible: {str(e)}")

@app.post("/recommend")
async def recommend_dineout(request: RecommendationRequest):
    try:
        print(f"Request recibido: {request.dict()}")  # DEBUG
        return get_recommendation_from_gemini(request)
    except ValidationError as e:
        print(f"Validation Error: {e}")
        print(f"Validation Error details: {e.errors()}")  # DEBUG
        raise HTTPException(status_code=422, detail=f"Invalid request: {e.errors()}")

@app.get("/")
def read_root():
    return {"status": "Tootli AI Service is running"}

@app.post("/debug-recommend")
async def debug_recommend_dineout(request: dict):
    try:
        print(f"Debug Request: {request}")
        recommendation_request = RecommendationRequest(**request)
        return get_recommendation_from_gemini(recommendation_request)
    except Exception as e:
        print(f"Debug Error: {e}")
        return {"error": str(e), "received_data": request}