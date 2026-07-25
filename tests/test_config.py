"""
Placeholder for Phase 2+ testing framework.
Every module must include pytest tests before merging.
"""


def test_config_loader_exists():
    from xrfm.config.loader import ConfigLoader

    loader = ConfigLoader()
    assert loader.get("project.name") == "XR Foundation Model"
