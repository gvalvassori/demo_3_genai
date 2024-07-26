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

client = genappbuilder.SearchServiceClient()
serving_config = client.serving_config_path(
    project=APP_BUILDER_PROJECT_ID,
    location=APP_BUILDER_LOCATION,
    data_store=SEARCH_ENGINE,
    serving_config=SERVING_CONFIG_ID,
)

def search(search_query: str, max_size: int=3) -> list:
    request = genappbuilder.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=max_size,
        content_search_spec={
            "summary_spec":{
                "summary_result_count": 3
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

    references = []
    results = list(response.results)

    print("Loop")
    for result in results:
        print("Loop", n_results)
        references.append(generate_reference(result, n_results))
        n_results += 1
        if n_results > max_size:
            break

    return references