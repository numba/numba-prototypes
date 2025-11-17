import pytest
from llm_midend import global_time_recorder

@pytest.hookimpl()
def pytest_sessionfinish(session, exitstatus):
    print("\n\n")
    print(global_time_recorder.pprint())
