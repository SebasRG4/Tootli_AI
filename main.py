import os
import re
import json # Importar json para un mejor formato en el prompt
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

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
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_km: Optional[float] = None

    class Config:
        extra = "ignore"

class UserLocation(BaseModel):
    latitude: float
    longitude: float

class RecommendationRequest(BaseModel):
    user_query: str
    user_name: str
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    candidates: Optional[List[Candidate]] = None
    history: Optional[List[Message]] = None
    previous_candidate_ids: Optional[List[int]] = None
    user_location: Optional[UserLocation] = None

    class Config:
        extra = "ignore"

app = FastAPI()

def get_recommendation_from_gemini(request: RecommendationRequest):
    model = genai.GenerativeModel('gemini-2.5-flash') # User requested 2.5
    chat_history = []
    if request.history:
        for msg in request.history:
            role = "user" if msg.role == "user" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})

    chat = model.start_chat(history=chat_history)

    # Convertir filtros a un string legible, ignorando los vacíos.
    filters_str = "Ninguno"
    if request.filters:
        # Usamos json.dumps para un formato limpio y legible
        filters_str = json.dumps(request.filters, indent=2, ensure_ascii=False)

    # --- PROMPT MEJORADO CON CONTEXTO DE FILTROS ---
    prompt = f"""¡Hola! Eres Toot, un asistente amigable y experto en restaurantes.

El usuario {request.user_name} está buscando: "{request.user_query}"

Filtros aplicados por el usuario:
{filters_str}

REGLAS OBLIGATORIAS:
1.  SOLO puedes recomendar restaurantes de la lista "RESTAURANTES DISPONIBLES". No inventes nada.
2.  Tu respuesta debe ser conversacional y útil. Explica brevemente (1-2 frases por lugar) por qué tus recomendaciones coinciden con la búsqueda Y los filtros.
3.  Si el request incluye "previous_candidate_ids", el usuario está refinando su búsqueda. Debes recomendar ÚNICAMENTE restaurantes que estén tanto en los `previous_candidate_ids` como en tu nueva selección. Si no hay coincidencias, informa al usuario amablemente.
4.  Al final de tu respuesta, DEBES incluir el token `[RECOMENDACION_IDS: id1, id2, ...]`. Si no hay recomendaciones, usa `[RECOMENDACION_IDS:]`.
5.  NO incluyas los IDs en el texto de la conversación, solo en el token final.
6.  Si el usuario pregunta por cercanía, distancia o cuál le queda más cerca, usa el campo 'distance_km' de cada restaurante para responder con PRECISIÓN. Indica la distancia exacta en kilómetros. Ordena tus recomendaciones del más cercano al más lejano.
7.  Si el usuario hace preguntas de seguimiento (como cercanía, precio, horario), da respuestas DETALLADAS usando toda la información disponible de los restaurantes.
"""

    if request.candidates and len(request.candidates) > 0:
        prompt += f"\n\nRESTAURANTES DISPONIBLES ({len(request.candidates)} encontrados que ya cumplen los filtros):"
        for c in request.candidates:
            features = []
            if getattr(c, 'featured', None): features.append("⭐ Destacado")
            if getattr(c, 'serves_alcohol', None): features.append("🍷 Sirve alcohol")
            if getattr(c, 'rating', None) and c.rating >= 4.0: features.append(f"🌟 Rating: {c.rating}")
            if getattr(c, 'delivery_time', None): features.append(f"⏱️ {c.delivery_time}")

            features_str = " | ".join(features) if features else "Estándar"
            tags = getattr(c, 'tags', [])
            tags_str = " | ".join(tags) if tags else "Sin tags específicos"
            discount_info = getattr(c, 'discount_info', None)
            tipo_cocina = getattr(c, 'tipo_cocina', 'Cocina variada')
            distance_km = getattr(c, 'distance_km', None)

            prompt += f"""
- ID: {c.id} | {c.name}
  - Cocina: {tipo_cocina}
  - Precio aprox. para dos: ${c.avg_price_for_two}
  - Tags: {tags_str}
  - Características: {features_str}{f" | Descuento: {discount_info}" if discount_info else ""}
  - Dirección: {c.address}
  - Distancia del usuario: {f"{distance_km} km" if distance_km is not None else "No disponible"}"""

        prompt += f"\n\nANÁLISIS: Basado en la búsqueda '{request.user_query}' y los filtros, ¿cuáles de estos restaurantes son la mejor opción? Justifica tu elección."
    else:
        prompt += "\n\nNo se encontraron restaurantes que coincidan con todos los filtros y la búsqueda. Sugiere al usuario que intente con otros términos o que quite algunos filtros. Termina tu respuesta con [RECOMENDACION_IDS:]."

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
            text_response = re.sub(r'\s*\[RECOMENDACION_IDS:[^\]]*\]\s*', '', text_response).strip()

        # DEFENSA: asegurar que las ids devueltas pertenecen a los candidates
        valid_candidate_ids = [c.id for c in request.candidates] if request.candidates else []
        recommendation_ids = [rid for rid in recommendation_ids if rid in valid_candidate_ids]

        # Si previous_candidate_ids fue enviada, aplicar intersección
        if request.previous_candidate_ids:
            previous_ids = [int(x) for x in request.previous_candidate_ids]
            intersection = [rid for rid in recommendation_ids if rid in previous_ids]
            
            # Si hay intersección, esos son los resultados. Si no, la IA ya debería haber generado un mensaje de fallback.
            recommendation_ids = intersection


        return {"responseText": text_response, "recommendation_ids": recommendation_ids}
    except Exception as e:
        print(f"Error al llamar a la API de Gemini: {e}")
        raise HTTPException(status_code=503, detail=f"El servicio de IA (Gemini) no está disponible: {str(e)}")

@app.post("/recommend")
async def recommend_dineout(request: RecommendationRequest):
    try:
        print(f"Request recibido: {request.dict()}")
        return get_recommendation_from_gemini(request)
    except ValidationError as e:
        print(f"Validation Error: {e.errors()}")
        raise HTTPException(status_code=422, detail=f"Invalid request: {e.errors()}")

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/get-embedding")
async def get_embedding(request: EmbeddingRequest):
    try:
        # Generate embedding using the "models/gemini-embedding-001" model
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=request.text,
            task_type="retrieval_document"
        )
        return {"embedding": result['embedding']}
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]

@app.post("/get-embeddings-batch")
async def get_embeddings_batch(request: BatchEmbeddingRequest):
    try:
        # Generate embeddings in batch
        # GEMINI supports list of strings for 'content'
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=request.texts,
            task_type="retrieval_document"
        )
        # result['embedding'] will be a list of lists (vectors)
        return {"embeddings": result['embedding']}
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Batch Embedding Error: {str(e)}")

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