# Adding Authentication

This guide explains how to add authentication to the LangGraph OpenAI Serve
OpenAI-compatible API while preserving compatibility with OpenAI clients. The
default API prefix is `/v1`; set `LGOS_OPENAI_API_PREFIX` or pass `prefix=` to
`bind_openai_chat_completion()` to mount it elsewhere.

## Why Add Authentication?

By default, LangGraph OpenAI Serve doesn't enforce authentication. This is fine for local development or internal use, but for production environments, you should implement authentication to:

- Prevent unauthorized access to your OpenAI-compatible API
- Track usage by different OpenAI-compatible clients
- Apply rate limits or quotas to specific users
- Control access to different graph models

## Simple API Key Authentication

Use the standard `Authorization: Bearer <api-key>` header for API keys. The
official OpenAI SDKs, Chainlit, Open WebUI, and most OpenAI-compatible clients
send the configured API key this way, so this preserves the OpenAI client
contract.

Here's how to implement a simple API key authentication system using FastAPI dependencies:

### 1. Create an Authentication Module

Create a file named `auth.py` with the following content:

```python
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.status import HTTP_401_UNAUTHORIZED

# OpenAI clients send api_key as Authorization: Bearer <key>.
bearer_scheme = HTTPBearer(auto_error=False)

# In a real application, store API keys in a database or environment variables
API_KEYS = {
    "valid_user_1": "sk-valid-key-1",
    "valid_user_2": "sk-valid-key-2",
}

async def get_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = credentials.credentials

    # Check if API key is valid
    for user, key in API_KEYS.items():
        if api_key == key:
            # Optionally return user info for logging
            return {"user": user, "key": api_key}

    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
```

### 2. Integrate Authentication with FastAPI App

Now update your application to use this authentication:

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED

from langgraph_openai_serve import LangchainOpenaiApiServe
from langgraph_openai_serve.core.settings import settings
from auth import API_KEYS

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_prefix = settings.OPENAI_API_PREFIX
        public_paths = {"/health", f"{api_prefix}/health"}
        if request.url.path in public_paths or not request.url.path.startswith(
            api_prefix
        ):
            return await call_next(request)

        authorization = request.headers.get("Authorization", "")
        scheme, _, api_key = authorization.partition(" ")
        if scheme.lower() != "bearer" or api_key not in API_KEYS.values():
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)

# Create a FastAPI app
app = FastAPI(
    title="Secure LangGraph API",
    description="LangGraph API with API key authentication",
)
app.add_middleware(APIKeyMiddleware)

# Initialize the LangGraph OpenAI Serve
graph_serve = LangchainOpenaiApiServe(
    app=app,
    graphs={
        "my_graph": my_graph,
    },
)

# Bind the OpenAI-compatible endpoints at settings.OPENAI_API_PREFIX.
graph_serve.bind_openai_chat_completion()
```

### 3. Using the OpenAI-Compatible API with Authentication

Once authentication is enabled, clients need to provide the API key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-valid-key-1"  # A valid API key is now required
)

response = client.chat.completions.create(
    model="my_graph",
    messages=[
        {"role": "user", "content": "Hello, how can you help me today?"}
    ]
)
```

## Advanced Authentication Options

For more robust authentication, consider these approaches:

### OAuth2 with JWT

For more sophisticated applications, you might want to implement OAuth2 with JWT:

```python
from datetime import datetime, timedelta
from typing import Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Secret key for JWT
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User database (in a real app, this would be a database)
fake_users_db = {
    "user@example.com": {
        "username": "user@example.com",
        "hashed_password": pwd_context.hash("userpassword"),
        "disabled": False,
    }
}

# Token and user models
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    disabled: Union[bool, None] = None

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, fake_db[username]["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# App setup with authentication
app = FastAPI()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
```

### Using Authentication Middleware

For a more global approach, you can use middleware. Keep the middleware on the
same bearer-token contract so OpenAI-compatible clients continue to work
without custom headers:

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED

from langgraph_openai_serve.core.settings import settings

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Exclude certain paths from authentication (like health checks)
        api_prefix = settings.OPENAI_API_PREFIX
        if request.url.path in {"/health", f"{api_prefix}/health"}:
            return await call_next(request)

        # OpenAI clients send api_key as Authorization: Bearer <key>.
        authorization = request.headers.get("Authorization", "")
        scheme, _, api_key = authorization.partition(" ")
        if (
            scheme.lower() != "bearer"
            or api_key not in ["sk-valid-key-1", "sk-valid-key-2"]
        ):
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content={"detail": "Invalid or missing API key"},
            )

        # Continue processing the request
        return await call_next(request)

app = FastAPI()
app.add_middleware(APIKeyMiddleware)
```

## Best Practices for API Authentication

1. **Use HTTPS**: Always use HTTPS in production to encrypt API keys and tokens in transit.

2. **Secure Storage**: Store API keys and user credentials securely (not in code).

3. **Key Rotation**: Implement a system to rotate API keys periodically.

4. **Scoped Access**: Consider implementing scopes to limit what different API keys can access.

5. **Rate Limiting**: Implement rate limiting based on API keys to prevent abuse.

6. **Logging**: Log authentication attempts for security auditing.

7. **Revocation**: Have a system to revoke API keys if they are compromised.

## Next Steps

- See [API reference](../reference.md) for detailed OpenAI-compatible endpoint
  documentation
