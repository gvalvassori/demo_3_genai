# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils_config import APP_BUILDER_PROJECT_ID, APP_BUILDER_LOCATION
from utils_config import SEARCH_ENGINE, SERVING_CONFIG_ID
from google.cloud import discoveryengine_v1beta as genappbuilder
from utils_crawler import generate_reference
from utils_vertex_llm import llm_predict
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

client = genappbuilder.SearchServiceClient()
serving_config = client.serving_config_path(
    project=APP_BUILDER_PROJECT_ID,
    location=APP_BUILDER_LOCATION,
    data_store=SEARCH_ENGINE,
    serving_config=SERVING_CONFIG_ID,
)

generation_config = {
    "max_output_tokens": 2048,
    "temperature": 0.2,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

def generate(text1):
    vertexai.init(project="corebigenai", location="us-central1")
    model = GenerativeModel("gemini-1.0-pro-001")
    responses = model.generate_content(
        [text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    result_query = ""
    for response in responses:
        result_query += response.text
        print(response.text, end="")
    return result_query


def search(search_query: str, max_size: int=1) -> list:
    request = genappbuilder.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=max_size,
        content_search_spec={
            "summary_spec":{
                "summary_result_count": 1
            },
            "snippet_spec": {
                "max_snippet_count": 2
            },
            "extractive_content_spec": {
                "max_extractive_answer_count": 2,
                "max_extractive_segment_count": 1
            }
        }
    )
    response = client.search(request)
    n_results = 1

    prompt = f"""Considere la tabla llamada `corebigenai.demos.productos`
                con el esquema que se describe a continuación
                Column name   Description
                prod_id      Codigo del Producto (Numero de Producto que identifica el producto)
                prod_name      Nombre del producto (Todo en letra minuscula)
                prod_name_long      Descripcion del producto (Todo en letra minuscula)
                prod_brand     Marca del producto (Todo en letra minuscula)
                subcategory    Subgcategoria del producto (Todo en letra minuscula)
                prod_unit_price      Precio unitario del producto
                prod_units   Tipo de unidades del producto (Todo en letra minuscula)
                prod_source  Proveedor o Fabricante (Todo en letra minuscula)

                Utilice siempre minúsculas si están buscando con la desccripcion o nombre del producto
                Sea optimo con las busquedas, trate de no llamar todas las variables, solo las necesarias
                prod_id es un numero

                Como experto en análisis de datos, escriba una consulta SQL en bigquery y siempre utilice un LIMIT 10 o menos.
                La solicitud es la siguiente solicitud: {search_query}"""
    result_query_2 = generate(prompt)
    result_query_2 = result_query_2.replace("\n", " ")
    result_query_2 = result_query_2.strip("`")
    result_query_2 = result_query_2.strip()
    result_query_2 = result_query_2.replace("sql", "")
    results = list(response.results)
    references = generate_reference(result_query_2, n_results)

    return references