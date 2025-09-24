#!/bin/bash

if [ ! -e SYNTAX_CHECKER_ENV ]; then
    echo "Error: SYNTAX_CHECKER_ENV file not found"
    exit 1
fi
if [ ! -e ../.env ]; then
    echo "Error: ../.env file not found"
    exit 1
fi

source SYNTAX_CHECKER_ENV
source ../.env

if [ -z "${GH_TOKEN+x}" ]; then
    echo "Error: GH_TOKEN variable is not defined."
    exit 1
fi
if [ -z "${GROUP_NUM+x}" ]; then
    echo "Error: GROUP_NUM variable is not defined."
    exit 1
fi
if [ -z "${1+x}" ]; then
    echo "Error: Log file not provided. Usage: ./get-log.sh <path-to-log>"
    exit 1
fi

curl --location --request GET 'http://dl-berlin.ecn.purdue.edu/api/log/download' \
--header 'Content-Type: application/json' \
--data "{
    \"group\": \"$GROUP_NUM\",
    \"gh_token\": \"$GH_TOKEN\",
    \"log\": \"$1\"
}"
