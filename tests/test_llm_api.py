from ece461.API import llm_api
import dotenv

def test_query_llm() -> None:
    dotenv.load_dotenv()
    response = llm_api.query_llm("What is the best software engineering tool?")
    assert isinstance(response, str)
    assert len(response) > 0
