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
if [ -z "${REPO_LINK+x}" ]; then
    echo "Error: REPO_LINK variable is not defined."
    exit 1
fi

curl --location 'http://dl-berlin.ecn.purdue.edu/api/register' \
--header 'Content-Type: application/json' \
--data "{
    \"group\": \"$GROUP_NUM\",
    \"github\": \"$REPO_LINK\",
    \"names\": [
            \"Joshua LeBlanc\",
	    \"Herbert de Bruyn\",
	    \"James Neff\",
	    \"Dylan Brannick\"
    ],
    \"gh_token\": \"$GH_TOKEN\"
}"
