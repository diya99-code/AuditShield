"""
FastAPI application for AP-Resolve (HF Space deployment).

Standalone version using absolute imports — works when the ap_resolve_env
directory is the root of the Space (no parent package).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the env root is on the path so absolute imports work
_env_root = str(Path(__file__).parent.parent)
if _env_root not in sys.path:
    sys.path.insert(0, _env_root)

from openenv.core.env_server.http_server import create_app

from models import APAction, APObservation
from server.ap_environment import APEnvironment

# On HF Spaces (SPACE_ID is set), enable web interface by default
_on_hf_space = bool(os.environ.get("SPACE_ID"))
_web_enabled = (
    os.environ.get("ENABLE_WEB_INTERFACE", "true" if _on_hf_space else "false").lower()
    == "true"
)

if _web_enabled:
    try:
        from openenv.core.env_server.web_interface import create_web_interface_app

        app = create_web_interface_app(
            APEnvironment,
            APAction,
            APObservation,
            env_name="ap_resolve_env",
        )
    except ImportError:
        app = create_app(APEnvironment, APAction, APObservation, env_name="ap_resolve")
else:
    app = create_app(APEnvironment, APAction, APObservation, env_name="ap_resolve")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
