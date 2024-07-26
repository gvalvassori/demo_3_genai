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

from utils_answer import answer_infobot, answer_conversational
from utils_dialogflow import process_infobot_webhook_request
from utils_dialogflow import process_conversational_webhook_request
from utils_dialogflow import json_to_webhook_request
from utils_dialogflow import format_infobot_webhook_response
from utils_dialogflow import format_conversational_webhook_response
import functions_framework
from flask import Request

@functions_framework.http
def dialogflow_request(request: Request):
    request_json = request.get_json(silent=True)
    if request_json and "fulfillmentInfo" in request_json:
        webhook_request = json_to_webhook_request(request_json)
        tag = webhook_request.fulfillment_info.tag
        
        if tag == "Infobot":
            infobot_args = process_infobot_webhook_request(
                webhook_request)
            response, search_results = answer_infobot(infobot_args)
            webhook_response = format_infobot_webhook_response(
                response, search_results, infobot_args)
            return webhook_response
        elif tag == "Conversational":
            conversational_args= process_conversational_webhook_request(
                webhook_request)
            response, history = answer_conversational(conversational_args)
            webhook_response = format_conversational_webhook_response(
                response, history)
            return webhook_response
    return "Hello"
