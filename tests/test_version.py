from langgraph_openai_serve.core import version as version_module


def test_get_version_reads_installed_package_metadata(monkeypatch):
    version_module.get_version.cache_clear()
    monkeypatch.setattr(
        version_module,
        "metadata_version",
        lambda package_name: f"metadata:{package_name}",
    )

    try:
        assert version_module.get_version() == "metadata:langgraph_openai_serve"
    finally:
        version_module.get_version.cache_clear()
