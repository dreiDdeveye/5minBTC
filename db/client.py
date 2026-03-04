"""Lightweight Supabase REST client using httpx (no supabase-py dependency)."""
import httpx
import config

_headers: dict | None = None
_base_url: str = ""


def _get_headers() -> dict:
    global _headers
    if _headers is None:
        _headers = {
            "apikey": config.SUPABASE_KEY,
            "Authorization": f"Bearer {config.SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
    return _headers


def _url(table: str) -> str:
    return f"{config.SUPABASE_URL}/rest/v1/{table}"


def select(table: str, columns: str = "*", params: dict | None = None) -> list[dict]:
    """SELECT from table with optional query params."""
    url = _url(table)
    query = {"select": columns}
    if params:
        query.update(params)
    resp = httpx.get(url, headers=_get_headers(), params=query, timeout=30)
    resp.raise_for_status()
    return resp.json()


def insert(table: str, data: dict | list[dict]) -> list[dict]:
    """INSERT row(s)."""
    url = _url(table)
    resp = httpx.post(url, headers=_get_headers(), json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def upsert(table: str, data: dict | list[dict], on_conflict: str = "") -> list[dict]:
    """UPSERT (INSERT ... ON CONFLICT)."""
    url = _url(table)
    headers = {**_get_headers(), "Prefer": "return=representation,resolution=merge-duplicates"}
    params = {}
    if on_conflict:
        params["on_conflict"] = on_conflict
    resp = httpx.post(url, headers=headers, json=data, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def update(table: str, data: dict, match: dict) -> list[dict]:
    """UPDATE rows matching conditions."""
    url = _url(table)
    params = {f"{k}": f"eq.{v}" for k, v in match.items()}
    resp = httpx.patch(url, headers=_get_headers(), json=data, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()
