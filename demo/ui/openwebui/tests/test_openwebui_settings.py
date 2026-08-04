from lgos_openwebui.settings import Settings


def test_settings_fields_have_descriptions() -> None:
    assert all(field.description for field in Settings.model_fields.values())
