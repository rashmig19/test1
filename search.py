def _default_distance_csv_path() -> str:
    """
    CSV path is configured from settings. Do not hardcode.
    Recommended: settings.DEFAULT_DISTANCE_FILEPATH points to 'Default Distance.csv'
    """
    path = getattr(settings, "DEFAULT_DISTANCE_FILEPATH", "") or ""
    if not path:
        raise RuntimeError("DEFAULT_DISTANCE_FILEPATH is not set in settings.")
    return path


@lru_cache(maxsize=1)
def _load_default_distance_rows() -> List[Dict[str, str]]:
    path = _default_distance_csv_path()
    if not os.path.exists(path):
        raise RuntimeError(f"Default distance CSV not found at: {path}")

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def get_default_radius_in_miles(group_id: str, practitioner_type: str = "PCP") -> int:
    """
    Looks up DefaultDistance from Default Distance.csv by:
      - Market == group_id
      - PractitionerType == 'PCP' (for assign PCP flow)
    Returns an int radius value.
    """
    rows = _load_default_distance_rows()

    gid = str(group_id).strip()
    ptype = str(practitioner_type).strip().lower()

    for r in rows:
        market = str(r.get("Market", "")).strip()
        prac = str(r.get("PractitionerType", "")).strip().lower()
        if market == gid and prac == ptype:
            dist = r.get("DefaultDistance", "")
            if dist is None or str(dist).strip() == "":
                break
            # DefaultDistance may be float-like; convert safely
            try:
                return int(float(str(dist).strip()))
            except Exception:
                raise RuntimeError(f"Invalid DefaultDistance '{dist}' for Market={gid}, PractitionerType={practitioner_type}")

    raise RuntimeError(f"No DefaultDistance found for Market={gid}, PractitionerType={practitioner_type}")


################################################

def provider_generic_search(
    group_id: str,
    subscriber_id: str,
    radius_in_miles: int,
    startingLocationZip: str,
    asOfDate: Optional[str] = None,
    providerLanguage: Optional[str] = None,
    offset: Optional[str] = None,
    providerSex: Optional[str] = None,
    networkStatus: Optional[str] = None,
    onlyPcps: Optional[str] = None,
    memberOverrideClass: Optional[str] = None,
    memberOverridePlan: Optional[str] = None,
    providerType: Optional[str] = None,
    sortType: Optional[str] = None,
    startingLocationAddr1: Optional[str] = None,
    limit: Optional[str] = None,
    memberId: Optional[str] = None,
    sortBy: Optional[str] = None,
    providerName: Optional[str] = None,
    serviceSpeciality: Optional[str] = None,
    prerProviderName: Optional[str] = None,
    prerPrprId: Optional[str] = None,
    prerPrprEntity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Calls Provider Generic Search REST API.
    Mandatory:
      - groupId, subscriberId, radiusInMiles, asOfDate
    ZIP-only flow uses:
      - startingLocationZip (user 5-digit ZIP)
      - radiusInMiles (from Default Distance.csv)
      - asOfDate (from user date else today's YYYYMMDD)
    """

    url = getattr(settings, "Generic_URL", "") or getattr(settings, "PROVIDER_GENERIC_URL", "")
    if not url:
        raise RuntimeError("Generic search URL not configured in settings (Generic_URL / PROVIDER_GENERIC_URL).")

    verify = settings.VERIFY_SSL_REST
    timeout = 30

    api_key = getattr(settings, "X_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    as_of_date = _format_as_of_date(asOfDate)

    # Defaults from settings when not provided (still not hardcoded)
    if limit is None:
        limit = getattr(settings, "limit", None)
    if offset is None:
        offset = getattr(settings, "offset", None)
    if onlyPcps is None:
        onlyPcps = getattr(settings, "onlyPcps", None)
    if memberOverrideClass is None:
        memberOverrideClass = getattr(settings, "memberOverrideClass", None)
    if memberOverridePlan is None:
        memberOverridePlan = getattr(settings, "memberOverridePlan", None)
    if startingLocationAddr1 is None:
        startingLocationAddr1 = getattr(settings, "startingLocationAddr1", "") or ""

    payload: Dict[str, Any] = {
        "groupId": group_id,
        "subscriberId": subscriber_id,
        "radiusInMiles": radius_in_miles,
        "asOfDate": as_of_date,
        "startingLocationZip": startingLocationZip,
        "startingLocationAddr1": startingLocationAddr1,

        # Optional parameters (only included if not None)
        "providerLanguage": providerLanguage,
        "offset": offset,
        "providerSex": providerSex,
        "networkStatus": networkStatus,
        "onlyPcps": onlyPcps,
        "memberOverrideClass": memberOverrideClass,
        "memberOverridePlan": memberOverridePlan,
        "providerType": providerType,
        "sortType": sortType,
        "limit": limit,
        "memberId": memberId,
        "sortBy": sortBy,
        "providerName": providerName,
        "serviceSpeciality": serviceSpeciality,
        "prerProviderName": prerProviderName,
        "prerPrprId": prerPrprId,
        "prerPrprEntity": prerPrprEntity,
    }

    payload = {k: v for k, v in payload.items() if v is not None}

    session = _session_with_retries()
    try:
        resp = session.post(url, headers=headers, json=payload, timeout=timeout, verify=verify)
        resp.raise_for_status()
        data = resp.json()
    except Exception as ex:
        raise RuntimeError(f"Provider generic search failed: {ex}") from ex

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

    return normalized

