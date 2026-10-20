"""Configured Pipes and an in-process Open WebUI admin API for tests."""

from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from lgos_openwebui.functions.generic.pipe import Pipe


def configured_pipe(pipe_type=Pipe, **valves):
    pipe = pipe_type()
    pipe.valves = pipe.Valves(
        **{
            "OPENAI_GATEWAY_TYPE": "litellm",
            "OPENAI_GATEWAY_BASE_URL": "http://lgos-litellm:4000",
            **valves,
        }
    )
    return pipe


ADMIN_TOKEN = "admin-token"


class FakeOpenWebUI(Starlette):
    """Open WebUI's admin routes, served in-process."""

    def __init__(
        self,
        functions: list[dict[str, Any]],
        *,
        password_auth: bool = True,
        signin_step: str | None = None,
        rejection: str | None = None,
    ) -> None:
        self.functions = {function["id"]: function for function in functions}
        self.password_auth = password_auth
        self.signin_step = signin_step
        self.rejection = rejection
        self.writes: list[str] = []
        self.signins = 0
        super().__init__(
            routes=[
                Route("/api/v1/auths/signin", self.signin, methods=["POST"]),
                Route("/api/v1/functions/export", self.export),
                Route("/api/v1/functions/id/{id}/valves", self.valves),
                Route("/api/v1/functions/{action:path}", self.write, methods=["POST"]),
            ]
        )

    async def signin(self, request: Request) -> JSONResponse:
        self.signins += 1
        if not self.password_auth:
            return JSONResponse({"detail": "Action prohibited"}, status_code=403)
        if await request.json() != {
            "email": "admin@example.com",
            "password": "password",
        }:
            return JSONResponse({"detail": "Invalid credentials"}, status_code=400)
        if self.signin_step is not None:
            # Open WebUI answers with the next step instead of a token.
            return JSONResponse({"next_step": self.signin_step})
        return JSONResponse({"token": ADMIN_TOKEN})

    async def export(self, request: Request) -> JSONResponse:
        if unauthorized := self.unauthorized(request):
            return unauthorized
        return JSONResponse(list(self.functions.values()))

    async def write(self, request: Request) -> JSONResponse:
        if unauthorized := self.unauthorized(request):
            return unauthorized
        self.writes.append(request.url.path)
        if self.rejection is not None:
            return JSONResponse({"detail": self.rejection}, status_code=400)
        action = request.path_params["action"]
        if action == "create":
            payload = await request.json()
            # Open WebUI lowercases the ID and creates the Function disabled.
            function_id = payload["id"].lower()
            self.functions[function_id] = {
                **payload,
                "id": function_id,
                "is_active": False,
            }
        elif action.endswith("/valves/update"):
            self.functions[action.split("/")[1]]["valves"] = await request.json()
        elif action.endswith("/toggle"):
            function = self.functions[action.split("/")[-2]]
            function["is_active"] = not function["is_active"]
        elif action.endswith("/update"):
            payload = await request.json()
            self.functions[payload["id"]].update(
                name=payload["name"], content=payload["content"], meta=payload["meta"]
            )
        return JSONResponse({})

    async def valves(self, request: Request) -> JSONResponse:
        if unauthorized := self.unauthorized(request):
            return unauthorized
        return JSONResponse(self.functions[request.path_params["id"]].get("valves"))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self),
            base_url="https://remote-webui.example",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        )

    @staticmethod
    def unauthorized(request: Request) -> JSONResponse | None:
        if request.headers.get("authorization") == f"Bearer {ADMIN_TOKEN}":
            return None
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
