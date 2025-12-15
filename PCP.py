
"""
Final fixed Horizon ChatModel compatible with LangChain / Pydantic v2.
"""

from __future__ import annotations

import os
import json
from typing import List, Any, Optional, Dict
from time import perf_counter
from contextvars import ContextVar
import urllib3
from dotenv import load_dotenv
from pydantic import PrivateAttr
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from .util import getAuthToken, sendHttpRequest
from config import settings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(".env_local")

_llm_telemetry: ContextVar[Optional[Dict[str, Any]]] = ContextVar("llm_telemetry", default=None)


def set_llm_telemetry(telemetry: Optional[Dict[str, Any]]):
    """Expose telemetry context so FastAPI handler can collect per-request LLM stats."""
    _llm_telemetry.set(telemetry)


def _estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    return len(str(value).split())


def _estimate_prompt_tokens(messages: List[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        total += _estimate_tokens(getattr(msg, "content", ""))
    return total


class ChatModel(SimpleChatModel):
    """Horizon-backed chat model compatible with LangChain."""

    # Properly declared Private Attributes
    _gateway: str = PrivateAttr(default=None)
    _client_id: str = PrivateAttr(default=None)
    _client_secret: str = PrivateAttr(default=None)
    _endpoint: str = PrivateAttr(default="/v2/text/chats")
    _qos: str = PrivateAttr(default="accurate")
    _reasoning: bool = PrivateAttr(default=True)
    _stream: bool = PrivateAttr(default=False)
    _timeout: Optional[float] = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        gateway: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        endpoint: str = "/v2/text/chats",
        qos: str = "accurate",
        reasoning: bool = True,
        stream: bool = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        # Always call parent init first
        super().__init__(**kwargs)

        # Assign values to declared private attrs
        object.__setattr__(self, "_gateway", settings.HORIZON_GATEWAY.rstrip("/"))
        object.__setattr__(self, "_client_id", client_id or settings.HORIZON_CLIENT_ID)
        object.__setattr__(self, "_client_secret", client_secret or settings.HORIZON_CLIENT_SECRET)
        object.__setattr__(self, "_endpoint", endpoint)
        object.__setattr__(self, "_qos", qos)
        object.__setattr__(self, "_reasoning", reasoning)
        object.__setattr__(self, "_stream", stream)
        object.__setattr__(self, "_timeout", timeout)

        if not self._gateway:
            raise ValueError("HORIZON_GATEWAY is not set.")
        if not self._client_id or not self._client_secret:
            raise ValueError("HORIZON_CLIENT_ID / HORIZON_CLIENT_SECRET must be set.")

    @property
    def _llm_type(self) -> str:
        return "horizon_chat"

    def _call(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """Send messages to Horizon Gateway and return assistant text."""
        telemetry = _llm_telemetry.get()
        prompt_tokens = _estimate_prompt_tokens(messages)
        completion_tokens = 0
        start = perf_counter()

        auth_token = getAuthToken(self._client_id, self._client_secret, self._gateway)
        if not auth_token:
            return "Error: Failed to acquire Horizon auth token."

        formatted_messages = []
        for msg in messages:
            role = (
                "user" if msg.type == "human"
                else "assistant" if msg.type == "ai"
                else "system" if msg.type == "system"
                else None
            )
            if role and getattr(msg, "content", None):
                formatted_messages.append({"role": role, "content": msg.content})

        if not formatted_messages:
            return "Error: No messages to send."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
        }
        body = {"messages": formatted_messages, "stream": self._stream}

        reasoning_str = "true" if self._reasoning else "false"
        endpoint = f"{self._endpoint}?qos={self._qos}&reasoning={reasoning_str}"

        try:
            resp = sendHttpRequest(
                data=json.dumps(body),
                header=headers,
                method="POST",
                address=self._gateway,
                endpoint=endpoint,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            msg = payload.get("message") or {}
            text = msg.get("content") or payload.get("content") or payload.get("text") or ""
            completion_tokens = _estimate_tokens(text)
            return text or "Error: Empty response from Horizon."
        except Exception as e:
            return f"API call failed: {e}"
        finally:
            if telemetry is not None:
                telemetry["llm_prompt_tokens"] = telemetry.get("llm_prompt_tokens", 0) + prompt_tokens
                telemetry["llm_completion_tokens"] = telemetry.get("llm_completion_tokens", 0) + completion_tokens
                telemetry["llm_latency_ms"] = telemetry.get("llm_latency_ms", 0.0) + (perf_counter() - start) * 1000


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = ChatModel()
    print(
        llm.invoke([
            SystemMessage(content="You are a test model."),
            HumanMessage(content="Ping from test.")
        ])
    )


#################################################################

import time
import requests
import logging
import os
from requests.adapters import HTTPAdapter, Retry
from app.config import settings
from app.llm.telemetry import (
    get_llm_telemetry,
    set_llm_telemetry
)
from app.observability.metrics import estimate_tokens
from time import perf_counter, sleep 
from typing import Dict, Any

logger = logging.getLogger("pcp_app")

# Timeouts & retry policy
DEFAULT_TIMEOUT = float(os.getenv("HORIZON_TIMEOUT_SECONDS", "30"))
RETRY_TOTAL = int(os.getenv("HORIZON_RETRY_TOTAL", "3"))
RETRY_BACKOFF = float(os.getenv("HORIZON_RETRY_BACKOFF", "0.5"))

verify_val = settings.CA_BUNDLE_PATH if settings.VERIFY_SSL_SOAP else False

# ------------------------------------------------------------------------------
# Token cache (memory)
# ------------------------------------------------------------------------------
_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,  # epoch seconds
}

# Simple in-memory token cache so we don’t call auth on every request
_member_token_cache: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,   # epoch seconds
}

def _auth_endpoint() -> str:
    """Build Horizon OAuth token endpoint."""
    if not settings.HORIZON_GATEWAY:
        raise ValueError("HORIZON_GATEWAY is not set")
    # Adjust if your Horizon auth path differs:
    return f"{settings.HORIZON_GATEWAY}/oauth2/token"

def _session_with_retries() -> requests.Session:
    """Requests session with basic retry policy suitable for gateways."""
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def call_horizon(system_prompt: str, user_prompt: str) -> str:   
    auth_token = getAuthToken(settings.HORIZON_CLIENT_ID, settings.HORIZON_CLIENT_SECRET, settings.HORIZON_GATEWAY)
    url = f"{settings.HORIZON_CHAT_ENDPOINT}"

    t0 = perf_counter()
    telemetry = get_llm_telemetry()
    if telemetry is not None:
        telemetry["llm_prompt_tokens"] = telemetry.get("llm_prompt_tokens", 0) + estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60, verify=verify_val)
    resp.raise_for_status()
    data=resp.json()
    # print(data)

    content = None
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")

            if content is None and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                m = choice.get("message") or {}
                content = m.get("content")
            if content is None and isinstance(data.get("text"), str):
                content = data["text"]

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    if isinstance(content, str):
        return content.strip()
    
    if isinstance(content, str):
        out_text = content.strip()
        if telemetry is not None:
            telemetry["llm_completion_tokens"] = telemetry.get("llm_completion_tokens", 0) + estimate_tokens(out_text)
            telemetry["llm_latency_ms"] = telemetry.get("llm_latency_ms", 0.0) + (perf_counter() - t0) * 1000
        return out_text
    
    if telemetry is not None:
        telemetry["llm_latency_ms"] = telemetry.get("llm_latency_ms", 0.0) + (perf_counter() - t0) * 1000
    
    return str(data)

def getAuthToken(client_id: str, client_secret: str, address: str) -> str:
    """
    Get (and cache) a Horizon Bearer token via client_credentials.
    - Respects in-memory cache until expiry (with a small safety margin).
    - Set env HORIZON_VERIFY_SSL to 'false' for dev self-signed, or to a CA bundle path.
    """
    # If a valid, non-expired token is cached, return it
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 10:
        return _token_cache["access_token"]

    # Validate inputs
    if not address:
        raise ValueError("Horizon address is required (HORIZON_GATEWAY).")
    if not client_id or not client_secret:
        raise ValueError("Horizon client_id/client_secret are required.")

    url = _auth_endpoint()
    # Most gateways accept standard OAuth2 client_credentials form body
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    
    session = _session_with_retries()
    try:
        resp = session.post(url, data=data, headers=headers, timeout=DEFAULT_TIMEOUT, verify=verify_val)
        resp.raise_for_status()
        payload = resp.json()

        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)  # seconds
        if not access_token:
            raise ValueError(f"Token endpoint missing 'access_token'. Response: {payload}")

        _token_cache["access_token"] = access_token
        _token_cache["expires_at"] = time.time() + float(expires_in)
        return access_token

    except requests.RequestException as e:
        logger.error("Horizon token request failed: %s", e, exc_info=True)
        raise
    except ValueError as e:
        logger.error("Invalid token response: %s", e, exc_info=True)
        raise
