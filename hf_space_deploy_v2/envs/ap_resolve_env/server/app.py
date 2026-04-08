"""
FastAPI application for AP-Resolve.

Creates the HTTP/WebSocket server using openenv_core's create_app helper.
Supports ENABLE_WEB_INTERFACE and TASK_ID environment variables.

On Hugging Face Spaces, ENABLE_WEB_INTERFACE defaults to true so the
Gradio playground is available at /web.
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from ..models import APAction, APObservation
from .ap_environment import APEnvironment

# On HF Spaces (SPACE_ID is set), enable web interface by default
_on_hf_space = bool(os.environ.get("SPACE_ID"))
_web_enabled = os.environ.get("ENABLE_WEB_INTERFACE", "true" if _on_hf_space else "false").lower() == "true"

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
        # Gradio not available, fall back to API-only
        app = create_app(APEnvironment, APAction, APObservation, env_name="ap_resolve")
else:
    app = create_app(APEnvironment, APAction, APObservation, env_name="ap_resolve")


def main() -> None:
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
