"""Navigation and session persistence for the main app."""
from .controller import AppController
from .delta_store import DeltaStore
from .registry import open_apps, register_app
from .session_store import SessionStore

__all__ = ["AppController", "DeltaStore", "SessionStore", "open_apps", "register_app"]
