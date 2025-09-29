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

    # Permitir campos extra para evitar errores 422
    class Config:
        extra = "ignore"

class RecommendationRequest(BaseModel):
    user_query: str
    user_name: str
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    candidates: Optional[List[Candidate]] = None
    history: Optional[List[Message]] = None

    # Permitir campos extra para evitar errores 422
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
    
    # PROMPT MEJORADO CON TAGS Y CARACTERÍSTICAS
    prompt = f"""Eres Toot, un asistente especializado en recomendaciones de restaurantes. Responde de forma conversacional pero perspicaz a {request.user_name}.

Consulta: "{request.user_query}"
Filtros aplicados: {request.filters if request.filters else 'Ninguno'}

INSTRUCCIONES:
- Analiza los TAGS y características únicas de cada restaurante
- Prioriza restaurantes que coincidan con la consulta mediante sus tags
- Destaca: características premium, descuentos, ratings altos, servicios especiales
- Sé específico y conciso (máximo 3-4 frases)
- Explica por qué estos restaurantes son buenas opciones

FORMATO OBLIGATORIO:
Al final incluir exactamente: [RECOMENDACION_IDS: id1, id2, ...]"""

    if request.candidates and len(request.candidates) > 0:
        prompt += f"\n\nRESTAURANTES DISPONIBLES ({len(request.candidates)} encontrados):"
        for c in request.candidates:
            # Construir características destacadas (con valores por defecto seguros)
            features = []
            if getattr(c, 'featured', None): features.append("⭐ Destacado")
            if getattr(c, 'serves_alcohol', None): features.append("🍷 Sirve alcohol")
            if getattr(c, 'rating', None) and c.rating >= 4.0: features.append(f"🌟 Rating: {c.rating}")
            if getattr(c, 'delivery_time', None): features.append(f"⏱️ {c.delivery_time}")
            
            features_str = " | ".join(features) if features else "Estándar"
            
            # Tags como elementos clave para el análisis
            tags = getattr(c, 'tags', [])
            tags_str = " | ".join(tags) if tags else "Sin tags específicos"
            
            discount_info = getattr(c, 'discount_info', None)
            tipo_cocina = getattr(c, 'tipo_cocina', 'Cocina variada')
            
            prompt += f"""
- ID: {c.id} | **{c.name}**
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

# Nuevo endpoint para debug
@app.post("/debug-recommend")
async def debug_recommend_dineout(request: dict):
    """Endpoint para debug que acepta cualquier JSON"""
    try:
        print(f"Debug Request: {request}")
        # Convertir el dict a RecommendationRequest
        recommendation_request = RecommendationRequest(**request)
        return get_recommendation_from_gemini(recommendation_request)
    except Exception as e:
        print(f"Debug Error: {e}")
        return {"error": str(e), "received_data": request}