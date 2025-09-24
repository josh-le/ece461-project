#!/bin/bash

if [ ! -e SYNTAX_CHECKER_ENV ]; then
    echo "Error: SYNTAX_CHECKER_ENV file not found"
    exit 1
fi
if [ -z "${GH_TOKEN+x}" ]; then
    if [ ! -e ../.env ]; then
	echo "Error: GH_TOKEN and ../.env file not found"
	exit 1
    fi
    source ../.env
fi

source SYNTAX_CHECKER_ENV

if [ -z "${GROUP_NUM+x}" ]; then
    echo "Error: GROUP_NUM variable is not defined."
    exit 1
fi

curl --location --request GET 'http://dl-berlin.ecn.purdue.edu/api/last_run' \
--header 'Content-Type: application/json' \
--data "{
    \"group\": \"$GROUP_NUM\",
    \"gh_token\": \"$GH_TOKEN\"
}"
