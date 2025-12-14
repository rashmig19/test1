# services/case_rest.py

import base64
import json
import logging
from typing import Any, Dict, Tuple

import requests
from config import settings

logger = logging.getLogger("uvicorn.error")


def _basic_auth_header(username: str, password: str) -> str:
    """
    Build a Basic Auth header value: 'Basic <base64(username:password)>'.
    """
    raw = f"{username}:{password}".encode("utf-8")
    token = base64.b64encode(raw).decode("ascii")
    return f"Basic {token}"


def _strip_px_keys(obj: Any) -> Any:
    """
    Recursively remove any dict keys that start with 'px'.
    Works for nested dicts and lists.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_px_keys(v)
            for k, v in obj.items()
            if not k.startswith("px")
        }
    if isinstance(obj, list):
        return [_strip_px_keys(v) for v in obj]
    return obj


def get_case(interaction_id: str) -> Dict[str, Any]:
    """
    Call the 'get case' endpoint using the interaction_id as a query parameter.
    Assumes:
      - URL in settings.GET_CASE_URL
      - Basic Auth creds in settings.CASE_BASIC_USERNAME / CASE_BASIC_PASSWORD
      - GET request with Content-Type / Accept: application/json
      - Query parameter name is 'interactionId' (change if your API uses a different name)
    """
    base_url = settings.GET_CASE_URL.rstrip("/")
    url = f"{base_url}%20{interaction_id}"
    print("get case url : ", url)
    if not url:
        raise RuntimeError("GET_CASE_URL is not configured")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": _basic_auth_header(
            settings.CASE_BASIC_USERNAME,
            settings.CASE_BASIC_PASSWORD,
        ),
    }

    verify = getattr(settings, "VERIFY_SSL_REST", True)

    logger.debug("Calling GET_CASE_URL=%s params=%s", url)
    resp = requests.get(
        url,
        headers=headers,
        timeout=30,
        verify=False if str(verify).lower() == "false" else verify,
    )
    logger.debug("get_case(%s) status=%s", interaction_id, resp.status_code)
    resp.raise_for_status()
    data = resp.json() if resp.text else {}
    print("status code get case : ", resp.status_code)
    # print("get case resp : ", data)
    return data


def create_case(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the 'create case' endpoint with the filtered payload (no px* keys).
    Assumes:
      - URL in settings.CREATE_CASE_URL
      - Basic Auth creds in settings.CASE_BASIC_USERNAME / CASE_BASIC_PASSWORD
      - POST request with JSON body
    """
    url = (getattr(settings, "CREATE_CASE_URL", "") or "").strip()
    if not url:
        raise RuntimeError("CREATE_CASE_URL is not configured")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": _basic_auth_header(
            settings.CASE_BASIC_USERNAME,
            settings.CASE_BASIC_PASSWORD,
        ),
    }

    verify = getattr(settings, "VERIFY_SSL_REST", True)

    logger.debug("Calling CREATE_CASE_URL=%s body=%s", url, json.dumps(payload)[:500])
    resp = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=30,
        verify=False if str(verify).lower() == "false" else verify,
    )

    logger.debug("create_case status=%s", resp.status_code)
    resp.raise_for_status()
    data = resp.json() if resp.text else {}
    # print("Create Case Response : ", data)
    return data


def sync_case_from_interaction(interaction_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function:
      1) Get case by interaction_id.
      2) Strip all 'px*' keys.
      3) Add AIInitiatedRequest under 'content'
      4) Call create_case with the cleaned payload.

    Returns: (raw_case, cleaned_case)
    """
    raw_case = get_case(interaction_id)
    cleaned_case = _strip_px_keys(raw_case)

    content: Dict[str, Any] = dict(cleaned_case)
    content.pop("caseTypeID", None)
    content["AIInitiatedRequest"] = "true"

    create_body: Dict[str, Any] = {
        "caseTypeID": settings.caseTypeID,
        "processID": settings.processID,
        "parentCaseID": "",
        "content": content,       # filtered (cleaned)  + AIInitiatedRequest
    }



    # Call create case API
    try:
        _ = create_case(create_body)
    except Exception as e:
        # We don't want this to break the PCP flow; log and re-raise if you prefer.
        logger.error("create_case failed for interaction_id=%s: %s", interaction_id, e, exc_info=True)
        # You can choose to re-raise if failure should stop the flow
        # raise
    print("Create Case Response : ", _)
    return raw_case, create_body
