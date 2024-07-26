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

from typing import List, Tuple
import vertexai
from utils_config import VERTEX_PROJECT_ID, VERTEX_LOCATION
from vertexai.preview.language_models import TextGenerationModel, ChatModel

vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)

def llm_predict(
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    content: str,
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)
    response = model.predict( # type: ignore
        content,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_k=top_k,
        top_p=top_p,)
    return response.text


def chat_predict(
    message: str,
    context: str,
    history: List[Tuple[str, str]],
    examples: list,
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int
    ) -> Tuple[str, List[Tuple[str, str]]]:
    """Predict using a Large Language Model."""

    chat_model = ChatModel.from_pretrained(model_name)
    parameters = {
      "temperature": temperature,
      "max_output_tokens": max_output_tokens,
      "top_p": top_p,
      "top_k": top_k,
    }

    chat_session = chat_model.start_chat( # type: ignore
        context=context, examples=examples)
    chat_session._history = history
    response = chat_session.send_message(message, **parameters)

    return response.text, chat_session._history