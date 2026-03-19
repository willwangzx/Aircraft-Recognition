from __future__ import annotations

from dataclasses import dataclass

from schemas.api import BatchPredictRequest, PredictRequest


@dataclass
class Response:
    status_code: int
    _payload: object

    def json(self) -> object:
        return self._payload


class TestClient:
    __test__ = False
    def __init__(self, app) -> None:
        self.app = app

    def post(self, path: str, json: dict | None = None) -> Response:
        handler = self.app.routes[("POST", path)]
        try:
            if path == "/predict":
                payload = handler(PredictRequest(**(json or {})))
            elif path == "/batch_predict":
                payload = handler(BatchPredictRequest(**(json or {})))
            else:
                payload = handler(json or {})
            return Response(status_code=200, _payload=payload)
        except Exception as exc:
            return Response(status_code=422, _payload={"detail": str(exc)})

    def get(self, path: str) -> Response:
        handler = self.app.routes[("GET", path)]
        payload = handler()
        return Response(status_code=200, _payload=payload)
