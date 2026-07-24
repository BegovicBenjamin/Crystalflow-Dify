from typing import Any

from dify_plugin import ToolProvider


class CrystalFlowProvider(ToolProvider):
    """CrystalFlow has no external credentials to validate."""

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        return
