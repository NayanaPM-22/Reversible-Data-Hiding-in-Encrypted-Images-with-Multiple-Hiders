import threading
import uuid
from typing import Optional

class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = {}
        return session_id

    def get(self, session_id: str) -> Optional[dict]:
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session_id: str, **kwargs):
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            self._sessions[session_id].update(kwargs)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def delete(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

store = SessionStore()