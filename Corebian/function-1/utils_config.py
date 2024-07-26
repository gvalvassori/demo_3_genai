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

import os

VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
APP_BUILDER_PROJECT_ID = os.environ.get("APP_BUILDER_PROJECT_ID", "")
APP_BUILDER_LOCATION = os.environ.get("APP_BUILDER_LOCATION", "global")
SEARCH_ENGINE = os.environ.get("SEARCH_ENGINE", "")
SERVING_CONFIG_ID = os.environ.get("SERVING_CONFIG_ID", "default_config")
ENGINE_TYPE = os.environ.get("ENGINE_TYPE", "")
