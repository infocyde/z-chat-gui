"""Point the app at a throwaway data directory *before* anything imports zchat.config."""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="zchat-test-"))
os.environ["ZCHAT_DATA_DIR"] = str(_tmp / "data")
os.environ["ZCHAT_IMAGE_DIR"] = str(_tmp / "image_output")
os.environ["ZCHAT_UPLOAD_DIR"] = str(_tmp / "uploads")
# Never hit the real API from tests, and don't depend on what's in the real .env.
os.environ["ZCHAT_ZAI_API_KEY"] = "test-key"
os.environ["ZCHAT_ZAI_BASE_URL"] = "https://api.z.ai/api/paas/v4"

import pytest  # noqa: E402

from zchat.db import upgrade_db  # noqa: E402

upgrade_db()  # builds the schema through the real Alembic migrations


@pytest.fixture
def s():
    from zchat.db import session_scope

    with session_scope() as session:
        yield session
