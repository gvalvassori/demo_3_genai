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

import json
from typing import List, Union, Tuple
from utils_answer import InfobotAnswerArgs, ConversationalAnswerArgs
from google.cloud.dialogflowcx_v3.types import WebhookRequest, WebhookResponse
from google.cloud.dialogflowcx_v3.types import response_message

REPLACE = WebhookResponse.FulfillmentResponse.MergeBehavior.REPLACE

def json_to_webhook_request(request_json:Union[dict, str, None]) -> WebhookRequest:
    if request_json:
        if isinstance(request_json, dict):
            request_json = json.dumps(request_json)
        if isinstance(request_json, str):
            return WebhookRequest.from_json(
                request_json, ignore_unknown_fields=True)
    
    return WebhookRequest()

def process_infobot_webhook_request(
        request: WebhookRequest) -> InfobotAnswerArgs:
    parameters = request.session_info.parameters
    if request.text:
        search_query = request.text
    elif request.transcript:
        search_query = request.transcript
    else:
        search_query = parameters.get("search_query", "")

    args = InfobotAnswerArgs(
        search_query=search_query,
        last_query=parameters.get("last_query", ""),
        context=parameters.get("context", ""),
        negative_response=parameters.get("negative_response", ""),
    )
    return args

def process_conversational_webhook_request(
        request: WebhookRequest) -> ConversationalAnswerArgs:
    parameters = request.session_info.parameters
    if request.text:
        message = request.text
    elif request.transcript:
        message = request.transcript
    else:
        message = parameters.get("search_query", "")
    args = ConversationalAnswerArgs(
        message=message,
        history=parameters.get("history", []),
        context=parameters.get("context", ""),
        examples=parameters.get("examples", [])
    )
    return args

def format_infobot_webhook_response(
        response: str,
        search_results: list,
        infobot_args: InfobotAnswerArgs) -> WebhookResponse:
    webhook_response = WebhookResponse()
    webhook_response.fulfillment_response.merge_behavior = REPLACE
    message_text = response_message.ResponseMessage()
    message_text.text = response_message.ResponseMessage.Text(text=[response])
    webhook_response.fulfillment_response.messages.append(message_text)
    message_rich = response_message.ResponseMessage()
    message_rich.payload = {
    "richContent": [
        [
        {
            "type": "info",
            "title": result.get("title"),
            "subtitle": result.get("snippet"),
            "actionLink": result.get("link")
        }
        ] for result in search_results
    ]
    }
    webhook_response.fulfillment_response.messages.append(message_rich)
    webhook_response.session_info.parameters["last_query"] = (
        f"Q:{infobot_args.search_query} "
        f"A:{response}"
    )
    return json.loads(WebhookResponse.to_json(webhook_response))

def format_conversational_webhook_response(
        response: str,
        history: List[Tuple[str, str]]) -> WebhookResponse:
    webhook_response = WebhookResponse()
    webhook_response.fulfillment_response.merge_behavior = REPLACE
    message_text = response_message.ResponseMessage()
    message_text.text = response_message.ResponseMessage.Text(text=[response])
    webhook_response.fulfillment_response.messages.append(message_text)
    webhook_response.session_info.parameters["history"] = history
    return json.loads(WebhookResponse.to_json(webhook_response))
