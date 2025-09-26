import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# --- CONFIGURACIÓN ---
# Carga la API key desde las variables de entorno de Render
try:
    # REEMPLAZA "GOOGLE_API_KEY" CON EL NOMBRE DE TU VARIABLE DE ENTORNO EN RENDER
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY'] 
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    # Si la variable no está, el servicio no iniciará (esto es bueno para detectar errores)
    raise RuntimeError("La variable de entorno GOOGLE_API_KEY no está configurada.")

# --- MODELOS DE DATOS (Pydantic) ---
# Estos no cambian, son los que tu backend de Laravel envía
class Candidate(BaseModel):
    id: int
    name: str
    address: str
    avg_price_for_two: float
    description: str

class RecommendationRequest(BaseModel):
    user_query: str
    user_name: str
    candidates: List[Candidate]

# --- INICIALIZACIÓN DE FASTAPI ---
app = FastAPI()

# --- LÓGICA DE LA IA CON GOOGLE GEMINI ---
def get_recommendation_from_gemini(request: RecommendationRequest):
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Creamos el prompt para Gemini (muy similar al que usabas para OpenAI)
    prompt_parts = [
        f"Analiza la lista de restaurantes candidatos basándote en la búsqueda del usuario: '{request.user_query}'."
        "Tu respuesta debe ser **muy breve, conversacional y amigable** (máximo 3 frases en total), ideal para un *snippet* de aplicación."
        "Recomienda **un único** restaurante que sea la mejor opción y justifica tu elección de forma concisa."
        "Si ninguno de los candidatos es una buena opción, indica amablemente que no se encontró un lugar ideal y sugiere intentar otra búsqueda."
        "\nCandidatos:"
    ]

    for c in request.candidates:
        prompt_parts.append(f"- ID: {c.id}, Nombre: {c.name}, Descripción: {c.description}, Precio Promedio: {c.avg_price_for_two}")
    
    prompt_parts.append("\nRespuesta del asistente:")

    try:
        # Hacemos la llamada a la API de Gemini
        response = model.generate_content(prompt_parts)
        
        # Devolvemos la respuesta en el formato que espera Laravel
        return {"responseText": response.text}

    except Exception as e:
        # Capturamos cualquier error de la API de Google
        print(f"Error al llamar a la API de Gemini: {e}")
        raise HTTPException(
            status_code=503, 
            detail="El servicio de IA (Gemini) no está disponible en este momento."
        )


# --- ENDPOINT DE LA API ---
@app.post("/recommend")
async def recommend_dineout(request: RecommendationRequest):
    # Aquí ya no hay límite de peticiones nuestro, solo llamamos a la función de Gemini
    return get_recommendation_from_gemini(request)

@app.get("/")
def read_root():
    return {"status": "Tootli AI Service is running"}
