import re
from mkdocs.plugins import BasePlugin


class MathExpression(BasePlugin):
    def on_page_markdown(self, markdown, **kwargs):
        # replace {{mathy:4x + 2y + x}} with "4x + 2y + x"
        markdown = re.sub(
            r"\{\{\s*mathy:([^\}]*)\s*\}\}", r"\1", markdown, flags=re.IGNORECASE
        )

        return markdown
