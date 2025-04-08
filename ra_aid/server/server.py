#!/usr/bin/env python3
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Callable, Any
import queue
import json

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Configure module-specific logging without affecting root logger
logger = logging.getLogger(__name__)
# Only configure this specific logger, not the root logger
if not logger.handlers:  # Avoid adding handlers multiple times
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.__stderr__)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Prevent propagation to avoid affecting the root logger configuration
    logger.propagate = False

# Add project root to Python path - Ensure this runs before other project imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ra_aid.server.api_v1_sessions import router as sessions_router
from ra_aid.server.api_v1_spawn_agent import router as spawn_agent_router
from ra_aid.server.connection_manager import ConnectionManager
from ra_aid.server.broadcast_sender import set_broadcast_queue

# Global variable to hold the app instance, needed by the hook (Now likely unused by hooks, but might be used elsewhere)
# This is a simple way, alternatives include passing app context differently.
_app_instance: FastAPI = None

async def broadcast_consumer(q: queue.Queue[Any], manager: ConnectionManager):
    """Consumes items (wrapped messages) from the queue, serializes the payload, and broadcasts the wrapper."""
    while True:
        try:
            wrapper = await asyncio.to_thread(q.get) # Assume item is the wrapper dict
            if not isinstance(wrapper, dict) or 'type' not in wrapper or 'payload' not in wrapper:
                 logger.warning(f"Received unexpected item format from broadcast queue: {type(wrapper)}. Expected {{'type': ..., 'payload': ...}}. Item: {str(wrapper)[:200]}...")
                 continue

            payload = wrapper['payload']
            message_type = wrapper['type']
            serializable_payload = None # Initialize for clarity

            try:
                # Check if the payload has a model_dump method (likely a Pydantic model)
                if hasattr(payload, 'model_dump') and callable(payload.model_dump):
                    # Convert Pydantic model to dict suitable for JSON
                    serializable_payload = payload.model_dump(mode='json')
                else:
                    # Assume payload is already JSON serializable
                    serializable_payload = payload

                # Update the payload within the wrapper
                wrapper['payload'] = serializable_payload

                # Attempt to serialize the entire wrapper
                message_str = json.dumps(wrapper)

            except TypeError:
                # Log if serialization fails even after potential payload conversion
                logger.warning(f"Could not JSON serialize wrapped message with type '{message_type}'. Payload type: {type(payload)}, Original Payload Preview: {str(payload)[:100]}... Wrapper Preview: {str(wrapper)[:200]}...")
                continue
            except Exception as e:
                 # Log other unexpected serialization errors
                 logger.error(f"Error during serialization of wrapper in broadcast_consumer: {e}. Wrapper Preview: {str(wrapper)[:200]}...")
                 continue

            # logger.debug(f"Broadcasting wrapped message of type: {message_type}")
            await manager.broadcast(message_str)
        except asyncio.CancelledError:
            logger.info("Broadcast consumer task cancelled.")
            break
        except Exception:
            logger.exception("Error in broadcast consumer task.")
            # Avoid breaking the loop on non-cancellation errors


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown events."""
    global _app_instance
    _app_instance = app # Store app instance globally

    logger.info("Application startup: Initializing resources.")
    # Store the running event loop
    app.state.loop = asyncio.get_running_loop()

    # Thread-safe queue for broadcasting messages to WebSocket clients.
    # Use app.state.broadcast_queue.put(item) from any thread.
    app.state.broadcast_queue = queue.Queue()
    set_broadcast_queue(app.state.broadcast_queue)

    # Instantiate the ConnectionManager
    app.state.connection_manager = ConnectionManager()
    # Start the consumer task
    app.state.broadcast_task = asyncio.create_task(
        broadcast_consumer(app.state.broadcast_queue, app.state.connection_manager)
    )

    yield  # Application is running

    logger.info("Application shutdown: Cleaning up resources.")

    # Cancel the consumer task
    if hasattr(app.state, 'broadcast_task') and app.state.broadcast_task:
        app.state.broadcast_task.cancel()
        try:
            await asyncio.wait_for(app.state.broadcast_task, timeout=5.0)
            logger.info("Broadcast consumer task successfully cancelled.")
        except asyncio.TimeoutError:
            logger.warning("Broadcast consumer task did not cancel within timeout.")
        except asyncio.CancelledError:
             logger.info("Broadcast consumer task already cancelled.") # Expected case
        except Exception:
            logger.exception("Error during broadcast consumer task cancellation.")

    # Clear global reference
    _app_instance = None
    logger.info("Application shutdown complete.")


# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="RA.Aid API",
    description="API for RA.Aid - AI Programming Assistant",
    version="1.0.0", # Consider fetching version dynamically
    lifespan=lifespan, # Add the lifespan context manager
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(sessions_router)
app.include_router(spawn_agent_router)

# Directory for the current file and prebuilt UI
CURRENT_DIR = Path(__file__).parent
PREBUILT_DIR = CURRENT_DIR / "prebuilt"
ASSETS_DIR = PREBUILT_DIR / "assets"
INDEX_HTML_PATH = PREBUILT_DIR / "index.html"

# Mount static assets directory if it exists
if ASSETS_DIR.exists() and ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
else:
    logger.warning(f"Assets directory not found or not a directory, skipping mount: {ASSETS_DIR}")

# WebSocket API endpoint using the ConnectionManager from app state
@app.websocket("/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager: ConnectionManager = websocket.app.state.connection_manager
    await manager.connect(websocket)
    logger.info(f"WebSocket client connected: {websocket.client}")
    try:
        # Keep the connection alive and detect disconnects by waiting for messages
        # This loop does nothing with the received data for now, just keeps the connection open
        while True:
            data = await websocket.receive_text()
            # logger.debug(f"Received message from {websocket.client}: {data}") # Optional: Log received messages
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {websocket.client}")
    except Exception as e:
         logger.error(f"WebSocket error for client {websocket.client}: {e}")
    finally:
        # Ensure disconnection happens even if errors occur within the loop
        manager.disconnect(websocket)
        logger.info(f"WebSocket connection closed for client: {websocket.client}")


@app.get("/", response_class=Response) # Changed response_class to base Response
async def get_root(request: Request) -> Response: # Changed return type annotation
    """Serve the prebuilt index.html or a fallback message."""
    if INDEX_HTML_PATH.exists() and INDEX_HTML_PATH.is_file():
        logger.debug(f"Serving index.html from: {INDEX_HTML_PATH}")
        return FileResponse(INDEX_HTML_PATH)
    else:
        logger.warning(f"index.html not found at: {INDEX_HTML_PATH}. Serving fallback HTML.")
        return HTMLResponse(
            '''
            <html>
                <head>
                    <title>RA.Aid API</title>
                    <style>
                        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
                    </style>
                </head>
                <body>
                    <h1>RA.Aid Server</h1>
                    <p>Web UI not built or not found. Run 'npm run build:web' in 'frontend/' directory.</p>
                    <p>A WebSocket API is available at /v1/ws for real-time updates (e.g., trajectory events).</p>
                    <p>See the <a href="/docs">API documentation</a> for more information on REST endpoints.</p>
                </body>
            </html>
            ''',
            status_code=200
        )


@app.get("/config")
async def get_config_endpoint(request: Request): # Renamed to avoid conflict with imported get_config
    """Return server configuration including host and port."""
    # Use the imported get_config function if needed, or just return client info
    # app_config = get_config() # Example if you needed config values
    return {"host": request.client.host, "port": request.scope.get("server")[1]}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RA.Aid API",
        summary="RA.Aid API OpenAPI Spec",
        version="1.0.0", # Consider dynamic version
        description="RA.Aid's API provides REST endpoints for managing sessions and agents, and a WebSocket endpoint (/v1/ws) for real-time communication of events like new trajectories. The root endpoint serves the static web UI if available.",
        routes=app.routes,
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        contact={
            "name": "RA.Aid Team",
            "url": "https://github.com/ai-christianson/RA.Aid",
        }
    )
    # Modify schema if needed, e.g., add WebSocket info manually if desired
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
