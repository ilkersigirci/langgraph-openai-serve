"""The demo's Open WebUI sends this service the documents it extracts."""

import re
from pathlib import Path

from lgos_files_api.api.files.extraction import EXTRACTED_SUFFIXES

OPEN_WEBUI_SERVICE = Path(__file__).parents[2] / "docker" / "apps" / "openwebui.yml"


def test_open_webui_lists_exactly_the_extracted_file_types() -> None:
    # Open WebUI's setting repeats this service's list: a type only Open WebUI
    # lists fails its uploads, and one only this service lists never reaches it.
    setting = re.search(
        r"^\s*OPENAI_FILES_API_FILE_EXTENSIONS:\s*(\S+)\s*$",
        OPEN_WEBUI_SERVICE.read_text(),
        re.MULTILINE,
    )

    assert setting is not None
    extensions = setting.group(1).strip("'\"").split(",")
    assert {f".{extension}" for extension in extensions} == EXTRACTED_SUFFIXES
