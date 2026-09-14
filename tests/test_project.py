from rag_assistant.project import get_project_name


def test_get_project_name() -> None:
    assert get_project_name() == "AI Research & Technical Document Assistant"