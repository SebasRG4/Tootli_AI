import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

# --- Configuración ---
# Carga la API key desde las variables de entorno de Render.
# Es más seguro que tenerla escrita directamente en el código.
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicializa la aplicación FastAPI
app = FastAPI()

# --- Modelos de Datos (Pydantic) ---
# Define la estructura de los datos que esperamos recibir en la petición.
# FastAPI usará esto para validar automáticamente los datos de entrada.
class Candidate(BaseModel):
    name: str
    address: str | None = None
    avg_price_for_two: float | None = None
    rating: float | None = None
    description: str | None = None

class SearchRequest(BaseModel):
    user_query: str
    user_name: str
    candidates: list[Candidate]

# --- Endpoints de la API ---

@app.get("/")
def read_root():
    """ Endpoint raíz para verificar que el servicio está funcionando. """
    return {"status": "Tootli AI Assistant is running"}

@app.post("/recommend")
def get_recommendation(request: SearchRequest):
    """
    Recibe la consulta de un usuario y una lista de restaurantes candidatos,
    y devuelve una recomendación generada por IA.
    """
    
    # 1. Construye el "prompt" para la IA.
    # Este es el paso más importante. Le damos contexto, instrucciones claras y los datos
    # con los que debe trabajar.
    system_prompt = f"""
    Eres "Tootli", un asistente amigable y experto en restaurantes para la app Dine-Out.
    Tu personalidad es servicial, entusiasta y un poco divertida.
    
    El usuario '{request.user_name}' te ha hecho la siguiente petición: "{request.user_query}".

    He pre-seleccionado estos restaurantes de nuestra base de datos que podrían ser buenas opciones.
    Tus recomendaciones DEBEN basarse ÚNICAMENTE en esta lista:
    {request.candidates}

    Tu tarea es:
    1. Saludar al usuario por su nombre de una forma cálida.
    2. Analizar su petición (ej. aniversario, presupuesto, tipo de comida) y demostrar que la entendiste.
    3. Recomendar 1 o 2 de los restaurantes de la lista que mejor se ajusten a su petición, explicando por qué. Destaca sus puntos fuertes (ej. rating, descripción, dirección).
    4. Si ningún candidato parece bueno, sé honesto y dile que no encontraste una opción perfecta, pero sugiérele el que más se aproxime.
    5. NO inventes información que no esté en la lista (como promociones o características no mencionadas).
    6. Mantén la respuesta concisa, amigable y en un solo párrafo si es posible.
    """

    try:
        # 2. Llama a la API de OpenAI (modelo Chat Completions)
        response = client.chat.completions.create(
            model="gpt-4o",  # Usamos gpt-4o por ser rápido, económico y de alta calidad.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Por favor, genera la recomendación para {request.user_name}."}
            ],
            temperature=0.7,  # Un valor entre 0.5 y 0.8 da un buen balance entre creatividad y consistencia.
            max_tokens=300    # Limita la longitud de la respuesta para controlar costos y mantenerla concisa.
        )
        
        ai_response = response.choices[0].message.content

        # 3. Devuelve la respuesta generada por la IA
        return {"responseText": ai_response.strip()}

    except openai.APIConnectionError as e:
        print(f"Error de conexión con OpenAI: {e}")
        raise HTTPException(status_code=503, detail="No se pudo conectar con el servicio de IA.")
    except openai.RateLimitError as e:
        print(f"Límite de peticiones excedido: {e}")
        raise HTTPException(status_code=429, detail="Demasiadas peticiones al servicio de IA. Intenta más tarde.")
    except openai.APIStatusError as e:
        print(f"Error en la API de OpenAI: {e.status_code} - {e.response}")
        raise HTTPException(status_code=500, detail=f"Error interno del servicio de IA: {e.status_code}")
    except Exception as e:
        print(f"Un error inesperado ocurrió: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error inesperado en el asistente.")

# --- Comandos para ejecutar localmente (no usado por Render) ---
# Si quieres probar esto en tu propia computadora:
# 1. Instala las librerías: pip install fastapi "uvicorn[standard]" openai python-dotenv
# 2. Crea un archivo .env con tu OPENAI_API_KEY='sk-...'
# 3. Ejecuta en tu terminal: uvicorn main:app --reload