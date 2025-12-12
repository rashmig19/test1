from typing import List, Dict, Any, Optional
from config import settings
import datetime
import requests
from requests.adapters import HTTPAdapter, Retry

# (you already have _session_with_retries and _format_as_of_date and normalize_provider_detail)

def provider_search_by_name(
    last_name: str,
    group_id: str,
    subscriber_id: str,
    member_id: str,
    first_name: Optional[str] = None,
    provider_name: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    startingLocationZip: Optional[str] = None,
    memberOverrideClass: Optional[str] = None,
    memberOverridePlan: Optional[str] = None,
    startingLocationAddr: Optional[str] = None,
    limit: Optional[str] = None,
    offset: Optional[str] = None,
    asOfDate: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Call Provider Search By Name REST API.

    Request payload includes (as per your spec):
      - lastName (mandatory)
      - groupId (mandatory)
      - subscriberId (mandatory)
      - memberId
      - firstName
      - providerName
      - city
      - state
      - startingLocationZip (optional 5-digit; used also for ZIP filtering)
      - memberOverrideClass
      - memberOverridePlan
      - startingLocationAddr
      - limit
      - offset
      - asOfDate (mandatory for API; if None we default to today's date in YYYYMMDD)
    """

    # Fill defaults from settings if not provided
    if limit is None:
        limit = getattr(settings, "limit", None)
    if offset is None:
        offset = getattr(settings, "offset", None)
    if memberOverrideClass is None:
        memberOverrideClass = getattr(settings, "memberOverrideClass", None)
    if memberOverridePlan is None:
        memberOverridePlan = getattr(settings, "memberOverridePlan", None)
    if startingLocationAddr is None:
        # if your settings has a specific addr default, use that
        startingLocationAddr = getattr(settings, "startingLocationAddr", "") or ""
    if startingLocationZip is None:
        startingLocationZip = getattr(settings, "startingLocationZip", "") or ""

    # asOfDate – if not provided, use today's date in YYYYMMDD
    as_of_date = _format_as_of_date(asOfDate)

    url = getattr(settings, "By_Name_URL")
    timeout = 30
    verify = settings.VERIFY_SSL_REST

    api_key = getattr(settings, "X_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    payload: Dict[str, Any] = {
        "lastName": last_name,
        "groupId": group_id,
        "subscriberId": subscriber_id,
        "memberId": member_id,
        "firstName": first_name,
        "providerName": provider_name,
        "city": city,
        "state": state,
        "startingLocationZip": startingLocationZip,
        "memberOverrideClass": memberOverrideClass,
        "memberOverridePlan": memberOverridePlan,
        "startingLocationAddr": startingLocationAddr,
        "limit": limit,
        "offset": offset,
        "asOfDate": as_of_date,
    }

    # Drop None values to avoid sending nulls if the API is strict
    payload = {k: v for k, v in payload.items() if v is not None}

    session = _session_with_retries()
    print("Name search payload:", payload)

    try:
        resp = session.post(url, headers=headers, json=payload, timeout=timeout, verify=verify)
        resp.raise_for_status()
        data = resp.json()
        print("Name search raw response:", data)
    except Exception as ex:
        raise RuntimeError(f"Provider search by Name/City/State failed: {ex}") from ex

    raw_details = data.get("providerDetails")
    if raw_details is None:
        return []

    if isinstance(raw_details, dict):
        raw_details = [raw_details]

    normalized: List[Dict[str, Any]] = []
    for detail in raw_details:
        try:
            normalized.append(normalize_provider_detail(detail))
        except Exception:
            continue

    # ZIP filter: if user provided a 5-digit ZIP, only keep providers whose
    # first 5 digits of the returned zip match.
    if startingLocationZip:
        user_zip5 = str(startingLocationZip)[:5]
        filtered: List[Dict[str, Any]] = []
        for p in normalized:
            pzip = str(p.get("zip") or "")
            if pzip[:5] == user_zip5:
                filtered.append(p)
        normalized = filtered

    return normalized
