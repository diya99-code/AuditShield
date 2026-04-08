"""
FastAPI application for AP-Resolve.

Creates the HTTP/WebSocket server using openenv_core's create_app helper.
Supports ENABLE_WEB_INTERFACE and TASK_ID environment variables.

On Hugging Face Spaces, ENABLE_WEB_INTERFACE defaults to true so the
Gradio playground is available at /web.
"""

from __future__ import annotations

import os

print("🚀🚀 SERVER BOOTING FROM ROOT 🚀🚀")

from openenv.core.env_server.http_server import create_app

from envs.ap_resolve_env.models import APAction, APObservation
from envs.ap_resolve_env.env import APEnvironment

# On HF Spaces (SPACE_ID is set), enable web interface by default
_on_hf_space = bool(os.environ.get("SPACE_ID"))
_web_enabled = os.environ.get("ENABLE_WEB_INTERFACE", "true" if _on_hf_space else "false").lower() == "true"

if _web_enabled:
    try:
        from openenv.core.env_server.web_interface import create_web_interface_app
        def ap_ui_builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
            import gradio as gr
            with gr.Blocks() as interface:
                gr.Markdown(f"### 🎯 Task Selector\nSelect a difficulty level and click **Launch** to reset the environment with that task.")
                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=["easy_straight_through", "medium_mismatch", "hard_duplicate_partial"],
                        value="easy_straight_through",
                        label="Select Difficulty"
                    )
                    launch_btn = gr.Button("🚀 Launch Task", variant="primary")
                
                output_msg = gr.Markdown("Select a task to begin.")
                
                async def launch_task(task_id):
                    try:
                        await web_manager.reset_environment({"task_id": task_id})
                        return f"✅ **Successfully launched:** `{task_id}`. Return to the **Playground** tab to start your audit."
                    except Exception as e:
                        return f"❌ **Error:** {str(e)}"
                
                launch_btn.click(fn=launch_task, inputs=[task_dropdown], outputs=[output_msg])
            return interface

        app = create_web_interface_app(
            APEnvironment,
            APAction,
            APObservation,
            env_name="ap_resolve_env",
            gradio_builder=ap_ui_builder
        )
    except ImportError:
        # Gradio not available, fall back to API-only
        app = create_app(APEnvironment, APAction, APObservation, env_name="ap_resolve")
else:
    app = create_app(APEnvironment, APAction, APObservation, env_name="ap_resolve")


def main() -> None:
    """Entry point for running the server directly."""
    import uvicorn
    # Use 7860 as the new standard
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
