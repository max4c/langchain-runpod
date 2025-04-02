# Langchain-Runpod Update TODO

## References
- Standard Tests: @https://python.langchain.com/docs/contributing/how_to/integrations/standard_tests/
- Package Structure: @https://python.langchain.com/docs/contributing/how_to/integrations/package/
- Publishing: @https://python.langchain.com/docs/contributing/how_to/integrations/publish/

## Tasks
- [x] Create `todo.md`
- [x] **Repository Structure Analysis**
    - [x] Review code against LangChain integration standards
    - [x] Identify missing components or structural issues (e.g., standard test directories)
- [x] **Implementation Updates (`langchain_runpod/llms.py`)**
    - [x] Add `_acall` implementation
    - [x] Add `_astream` implementation
    - [x] Refactor `_process_response` for clarity and robustness
    - [x] Improve error handling (e.g., more specific exceptions)
    - [x] Update class and method docstrings
    - [x] Add polling logic for async jobs
- [x] **Testing Implementation**
    - [x] Create `tests/unit_tests` directory
    - [x] Create `tests/integration_tests` directory
    - [x] Add `pytest` and `langchain-tests` to dev dependencies
    - [x] Implement custom unit tests in `tests/unit_tests/test_llms.py`
    - [x] Implement custom integration tests in `tests/integration_tests/test_llms.py`
    - [x] Run unit tests (`pytest tests/unit_tests`)
    - [x] Run integration tests (`pytest tests/integration_tests`)
    - [x] Fix integration tests for `ChatRunPod`
- [x] **Package Publication**
    - [x] Update `pyproject.toml` metadata (description, authors, classifiers, etc.)
    - [x] Increment package version in `pyproject.toml`
- [x] **Final Review**
    - [x] Ensure all TODOs are checked
    - [x] Review changes for consistency and correctness 