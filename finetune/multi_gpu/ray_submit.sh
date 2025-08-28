#!/bin/bash

export TOKEN=LLM_TOKEN
export RAY_ADDRESS="https://localhost/cluster0/"
export RAY_TLS_CA_CERT="RAY_CA.pemfile"
export RAY_JOB_HEADERS="{\"Authorization\":\"Bearer $TOKEN\"}"

export RAY_JOB_RUNTIME_ENV='{
"pip": "requirement3.txt",
"env_vars": {
    "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1",
    "RAY_COLOR_PREFIX": "0",
    "ARTIFACT_STORAGE_ROOT": "/home/user/ray/storage_root",
    "ARTIFACT_STORAGE_USERNAME": "user"
  }
}'

export MY_RAY_JOB_METADATA='{
"EID": "user",
"E-mail": "user@gmail.com",
"Description": "Regular training task."
}'

echo $RAY_JOB_RUNTIME_ENV

timestamp=$(date +"%Y%m%d%H%M%S")

ray job submit \
--log-color=false \
--log-style=record \
--no-wait \
--address=$RAY_ADDRESS \
--verify=$RAY_TLS_CA_CERT \
--headers="$RAY_JOB_HEADERS" \
--metadata-json="$RAY_JOB_METADATA" \
--runtime-env-json="$RAY_JOB_RUNTIME_ENV" \
--working-dir './' \
-- python3 -m sft_conversation_training -t "$timestamp"
