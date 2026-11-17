import io

import docx
import pytest

from zchat import documents


async def test_text_file_is_extracted():
    p = await documents.process_upload("../../evil notes.md", b"# Title\nhello")
    assert p.kind == "document" and p.text == "# Title\nhello"
    assert p.filename == "evil notes.md"  # path parts stripped
    assert "evil" not in p.path  # stored under a generated name


async def test_docx_is_extracted():
    d = docx.Document()
    d.add_paragraph("Chapter one")
    buf = io.BytesIO()
    d.save(buf)
    p = await documents.process_upload("story.docx", buf.getvalue())
    assert "Chapter one" in p.text


async def test_image_is_stored_without_text():
    p = await documents.process_upload("cat.PNG", b"\x89PNG")
    assert p.kind == "image" and p.text == "" and p.mime == "image/png"


async def test_unsupported_file_type():
    with pytest.raises(documents.UnsupportedFile):
        await documents.process_upload("tool.exe", b"MZ")
