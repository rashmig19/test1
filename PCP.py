# provider_search.py

import requests
from typing import List, Dict, Any
from config import settings
from requests.adapters import HTTPAdapter, Retry


# ---------------------------
# Utility: create session with retries
# ---------------------------
def _session_with_retries() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------
# Normalize provider response
# ---------------------------
def normalize_provider_detail(detail: dict) -> dict:
    info = detail.get("providerInfo") or {}
    contact = detail.get("providerContact") or {}

    return {
        "providerId": info.get("providerId"),
        "name": info.get("providerName") or info.get("providerFullName"),
        "address1": contact.get("addressLine1"),
        "address2": contact.get("addressLine2"),
        "city": contact.get("city"),
        "state": contact.get("state"),
        "zip": contact.get("zip"),
        "county": contact.get("county"),
        "phone": contact.get("phone"),
        "fax": contact.get("fax"),
        "networkStatus": info.get("networkStatus"),
        "isAcceptingNewMembers": info.get("isAcceptingNewMembers"),
        "pcpAssnInd": info.get("pcpAssnInd"),
        "distance_mi": info.get("distanceInMiles"),
    }


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def provider_search_by_id(provider_id: str) -> List[Dict[str, Any]]:
    """
    Calls the Provider Search By ID REST API endpoint,
    extracts providerDetails → providerInfo & providerContact,
    normalizes them, and returns a list of provider records.

    Returns:
        List[Dict]: Normalized provider records.
    """

    url = settings.PROVIDER_SEARCH_BY_ID_URL
    timeout = settings.REST_TIMEOUT_SECONDS
    verify = settings.VERIFY_SSL_REST

    payload = {
        "providerId": provider_id,
        "requestFrom": settings.REQUEST_FROM,
        "distanceInMiles": settings.DEFAULT_DISTANCE_MILES,
        "homeZipCode": settings.DEFAULT_ZIP,
        "groupId": settings.DEFAULT_GROUP_ID,
    }

    session = _session_with_retries()

    try:
        resp = session.post(url, json=payload, timeout=timeout, verify=verify)
        resp.raise_for_status()
        data = resp.json()

    except Exception as ex:
        raise RuntimeError(f"Provider search by ID failed: {ex}") from ex

    # ---------------------------
    # VALIDATE RESPONSE STRUCTURE
    # ---------------------------
    raw_details = data.get("providerDetails")

    if raw_details is None:
        # API might return single object, list, or empty
        return []

    # Ensure it's a list
    if isinstance(raw_details, dict):
        raw_details = [raw_details]

    # ---------------------------
    # NORMALIZE EACH ENTRY
    # ---------------------------
    normalized: List[Dict[str, Any]] = []

    for detail in raw_details:
        try:
            normalized.append(normalize_provider_detail(detail))
        except Exception:
            # skip malformed entries but do not break function
            continue

    return normalized
