from ece461.API import llm_api

def test_query_llm() -> None:
    response = llm_api.query_llm("What is the best software engineering tool?")
    assert isinstance(response, str)
    assert len(response) > 0