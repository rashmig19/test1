# provider_search.py

def normalize_provider_detail(detail: dict) -> dict:
    info = detail.get("providerInfo") or {}
    contact = detail.get("providerContact") or {}

    flat = {
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
    return flat


def provider_search_by_id_normalized(provider_id: str) -> list[dict]:
    raw = provider_search_by_id(provider_id=provider_id)

    # handle your real response shape
    if isinstance(raw, dict) and "providerDetails" in raw:
        details = raw["providerDetails"]
        if isinstance(details, dict):
            details = [details]
        return [normalize_provider_detail(d) for d in details]

    # fallback to old behavior if needed
    if isinstance(raw, list):
        return raw
    return [raw] if raw else []
