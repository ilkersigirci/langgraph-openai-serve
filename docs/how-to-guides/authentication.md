# Authentication

The package does not enforce authentication by default. Add it at the FastAPI
host layer while preserving the OpenAI client contract:

- use `Authorization: Bearer <api-key>`
- avoid custom auth headers for OpenAI-compatible routes
- keep health checks public if your platform needs unauthenticated probes

## API Key Middleware

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED

from langgraph_openai_serve import LanggraphOpenaiServe
from langgraph_openai_serve.core.settings import settings

API_KEYS = {"sk-valid-key-1", "sk-valid-key-2"}

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_prefix = settings.OPENAI_API_PREFIX
        if request.url.path in {"/health", f"{api_prefix}/health"}:
            return await call_next(request)
        if not request.url.path.startswith(api_prefix):
            return await call_next(request)

        scheme, _, api_key = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or api_key not in API_KEYS:
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)

app = FastAPI()
app.add_middleware(APIKeyMiddleware)
LanggraphOpenaiServe(app=app, graphs=graphs).bind_openai_api()
```

Clients then pass the key through normal OpenAI SDK configuration:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-valid-key-1")
```

## Production Notes

- Store keys and credentials outside source code.
- Use HTTPS.
- Add rate limits, usage logging, key rotation, and revocation.
- OAuth2/JWT can work if the resulting access token is still supplied as a
  bearer token to OpenAI-compatible clients.
