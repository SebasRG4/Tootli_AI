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
    items: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    coupons: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

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
    model = genai.GenerativeModel('gemini-1.5-flash') # 15 req/min free tier vs 5 in 2.5-flash
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
8.  **OPCIÓN DE PRESUPUESTO (Smart Budget Matcher)**: Si el usuario indica un presupuesto máximo (ej. "Tengo $300"), analiza los precios del campo "Menú disponible" de los restaurantes candidatos y cruza la información con los "Cupones de descuento activos". Calcula combinaciones reales de platillos que quepan dentro del presupuesto del usuario aplicando el descuento correspondiente, explicando el cálculo y los precios con precisión matemática.
9.  **OPCIÓN DE RUTA / ITINERARIO (Gastronomic Routes)**: Si el usuario solicita una ruta gastronómica, tour, crawl, itinerario o recorrido de varios lugares (ej. "ruta de postres", "tour de tacos"), DEBES:
    a) Recomendar de 2 a 4 restaurantes candidatos en una secuencia lógica (ej. entrada, plato fuerte, postre, o cercanía).
    b) Añadir al final de tu respuesta el token `[ROUTE: true]`. De lo contrario, si es una búsqueda normal de lugares individuales, añade `[ROUTE: false]`.
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

            # Format items
            items_list = getattr(c, 'items', [])
            items_str = ", ".join([f"{item['name']} (${item['price']})" for item in items_list]) if items_list else "No disponible"

            # Format coupons
            coupons_list = getattr(c, 'coupons', [])
            coupons_str = ", ".join([f"Código {coupon['code']}: {coupon['title']} (Desc: {coupon['discount']} {coupon['discount_type']})" for coupon in coupons_list]) if coupons_list else "Sin cupones activos"

            prompt += f"""
- ID: {c.id} | {c.name}
  - Cocina: {tipo_cocina}
  - Precio aprox. para dos: ${c.avg_price_for_two}
  - Menú disponible (platillos destacados y precios): {items_str}
  - Cupones de descuento activos para este restaurante: {coupons_str}
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

        # Extraer bandera de ruta [ROUTE: true/false]
        is_route = False
        route_match = re.search(r'\[ROUTE:\s*(true|false)\]', text_response, re.IGNORECASE)
        if route_match:
            is_route = route_match.group(1).lower() == 'true'
            text_response = re.sub(r'\s*\[ROUTE:\s*(true|false)\]\s*', '', text_response, flags=re.IGNORECASE).strip()

        # DEFENSA: asegurar que las ids devueltas pertenecen a los candidates
        valid_candidate_ids = [c.id for c in request.candidates] if request.candidates else []
        recommendation_ids = [rid for rid in recommendation_ids if rid in valid_candidate_ids]

        # Si previous_candidate_ids fue enviada, aplicar intersección
        if request.previous_candidate_ids:
            previous_ids = [int(x) for x in request.previous_candidate_ids]
            intersection = [rid for rid in recommendation_ids if rid in previous_ids]
            
            # Si hay intersección, esos son los resultados. Si no, la IA ya debería haber generado un mensaje de fallback.
            recommendation_ids = intersection

        return {"responseText": text_response, "recommendation_ids": recommendation_ids, "is_route": is_route}
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

import base64

class MenuExtractRequest(BaseModel):
    image_base64: str
    mime_type: str

@app.post("/extract-menu")
async def extract_menu(request: MenuExtractRequest):
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(request.image_base64)
        
        # Prepare the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """Analiza la imagen de este menú de restaurante. Extrae todos los platillos, bebidas y postres con sus respectivos precios y descripciones.
        Debes responder estrictamente con un objeto JSON válido que tenga una propiedad llamada 'items' que contenga un arreglo de objetos. Cada objeto debe tener exactamente los siguientes campos:
        - 'name': Nombre del platillo o bebida (limpio y bien escrito).
        - 'description': Descripción del platillo o sus ingredientes (si no tiene descripción, genera una descripción corta apetitosa basada en su nombre).
        - 'price': Precio base como un número flotante o entero sin signos de pesos ni comas (ej. 150.00). Si hay varias opciones de precio según el tamaño, usa el precio de la opción más económica como precio base. Si no tiene precio, calcula un precio estimado promedio de 120.00.
        - 'suggested_category': El nombre de la categoría a la que pertenece (ej. 'Entradas', 'Platos Fuertes', 'Bebidas', 'Postres', 'Tacos', 'Pizzas').
        - 'available_time_starts': Hora de inicio en que se sirve el platillo en formato "HH:MM:SS" (ej. "08:00:00" si es un desayuno, de lo contrario por defecto "00:00:00").
        - 'available_time_ends': Hora de fin en que se sirve el platillo en formato "HH:MM:SS" (ej. "12:30:00" si es un desayuno, de lo contrario por defecto "23:59:59").
        - 'variations': Un arreglo de grupos de variantes/adicionales (opciones) si el platillo las tiene (ej. tamaños: "Chico $90, Grande $120", proteínas: "Pollo/Res", o extras: "Extra Queso +$15"). Si no tiene variantes, este arreglo debe ser un arreglo vacío []. Cada objeto de grupo en este arreglo debe tener exactamente esta estructura:
          * 'name': Nombre del grupo de variación (ej. "Tamaño", "Proteína", "Extras").
          * 'type': 'single' (si solo se puede seleccionar una opción) o 'multi' (si se pueden elegir múltiples opciones).
          * 'min': Entero (ej. 1 si es obligatorio y 'single', o 0 si es opcional).
          * 'max': Entero (ej. 1 si es 'single', o el número de opciones si es 'multi').
          * 'required': 'on' (si es obligatorio seleccionar al menos uno) o 'off' (si es opcional).
          * 'values': Un arreglo de objetos, donde cada objeto representa una opción del grupo:
            - 'label': Nombre de la opción (ej. "Chico", "Grande", "Pollo", "Extra Queso").
            - 'optionPrice': El costo ADICIONAL (como número flotante) con respecto al precio base del platillo (ej. si el platillo cuesta $90 base y la opción Grande cuesta $120, el optionPrice para "Chico" es 0.0 y para "Grande" es 30.0. Si la opción de extra es +$15, el optionPrice es 15.0).
        
        Responde únicamente en formato JSON estructurado, sin texto de introducción, sin bloques markdown de código ```json ... ```, solo el JSON puro."""
        
        image_data = {
            'mime_type': request.mime_type,
            'data': image_bytes
        }
        
        # Call Gemini model
        response = model.generate_content([image_data, prompt])
        text_response = response.text
        
        # Clean the output in case the model returns markdown code block
        text_response = text_response.strip()
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.startswith("```"):
            text_response = text_response[3:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
        text_response = text_response.strip()
        
        # Parse it to verify it is valid JSON
        data = json.loads(text_response)
        
        return data
        
    except Exception as e:
        print(f"Error in extract_menu: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting menu: {str(e)}")

from fastapi import File, UploadFile
from fastapi.responses import Response
from rembg import remove
from PIL import Image
import io

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents))
        
        # Run background removal
        output_image = remove(input_image)
        
        # Save output image to buffer as PNG (which supports transparency)
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
    except Exception as e:
        print(f"Error in remove_background: {e}")
        raise HTTPException(status_code=500, detail=f"Error removing background: {str(e)}")