import json
from typing import List, NamedTuple, Tuple
from utils_vertex_llm import llm_predict, chat_predict
from utils_app_builder import search
from vertexai.preview.language_models import InputOutputTextPair

class InfobotAnswerArgs(NamedTuple):
    search_query: str
    last_query: str
    context: str
    negative_response: str

class ConversationalAnswerArgs(NamedTuple):
    message: str
    history: List[Tuple[str, str]]
    context:str
    examples: List[InputOutputTextPair]

def answer_infobot(
        args: InfobotAnswerArgs):
    search_results = search(args.search_query)
    if not search_results.get('long_snippet'):
        return args.negative_response, search_results
    last_query_ = args.last_query if args.last_query else "Responde lo siguiente "
    search_result_text = search_results.get('long_snippet')
    print(last_query_)
    print(args.search_query)
    print(search_result_text)
    prompt = args.context.format(
        last_query=last_query_,
        search_query=args.search_query,
        search_results=search_result_text)
    print(prompt)
    response = llm_predict(
        model_name="text-bison@001",
        temperature=0.2,
        max_output_tokens=1024,
        top_p=0.8,
        top_k=40,
        content=prompt)
    return response, search_results.get('long_snippet')
def answer_conversational(
        args: ConversationalAnswerArgs):
    
    response = chat_predict(
        message=args.message,
        context=args.context,
        history=args.history,
        examples=args.examples,
        model_name="chat-bison@001",
        temperature=0.2,
        max_output_tokens=1024,
        top_p=0.8,
        top_k=40)

    return response