from __future__ import annotations

from .testclient import TestClient


class FastAPI:
    def __init__(self, title: str = "", version: str = "") -> None:
        self.title = title
        self.version = version
        self.routes: dict[tuple[str, str], callable] = {}

    def get(self, path: str):
        def decorator(func):
            self.routes[("GET", path)] = func
            return func
        return decorator

    def post(self, path: str):
        def decorator(func):
            self.routes[("POST", path)] = func
            return func
        return decorator
