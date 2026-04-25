"""Google Drive ingestion (optional dependency).

Soft-fails with a clear message if `google-api-python-client` and friends
are not installed. Install with: pip install -e ".[drive]"

OAuth flow follows Google's standard installed-app pattern. Only the
client_secret JSON path and the cached token path are configurable; both
defaults are gitignored.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ..config import load_config, load_sources
from ..wiki.knowledge_base import KnowledgeBase

log = logging.getLogger(__name__)


_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_EXPORT_MIMES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


def _build_service():
    try:
        from google.auth.transport.requests import Request  # type: ignore
        from google.oauth2.credentials import Credentials  # type: ignore
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
        from googleapiclient.discovery import build  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive deps missing. Install with: pip install -e \".[drive]\""
        ) from exc

    client_secrets = Path(
        os.getenv("RKE_DRIVE_CLIENT_SECRETS", "credentials.json")
    ).expanduser()
    token_cache = Path(
        os.getenv("RKE_DRIVE_TOKEN_CACHE", "google_token.json")
    ).expanduser()

    if not client_secrets.exists():
        raise RuntimeError(
            f"Missing OAuth client secrets at {client_secrets}. "
            "Download from GCP console → APIs & Services → Credentials."
        )

    creds = None
    if token_cache.exists():
        creds = Credentials.from_authorized_user_file(str(token_cache), _SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), _SCOPES)
            creds = flow.run_local_server(port=0)
        token_cache.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def list_folder(service, folder_id: str) -> list[dict]:
    files: list[dict] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=100,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def fetch_file_text(service, file_meta: dict) -> str | None:
    fid = file_meta["id"]
    mime = file_meta["mimeType"]
    name = file_meta["name"]
    try:
        if mime in _EXPORT_MIMES:
            data = service.files().export(fileId=fid, mimeType=_EXPORT_MIMES[mime]).execute()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return str(data)
        if mime.startswith("text/") or mime == "application/json":
            data = service.files().get_media(fileId=fid).execute()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return str(data)
        log.info("skipping %s (unsupported mime: %s)", name, mime)
        return None
    except Exception as exc:
        log.warning("fetch failed for %s: %s", name, exc)
        return None


def ingest_drive_folder(folder_id: str, category: str = "drive", tags: list[str] | None = None) -> int:
    service = _build_service()
    cfg = load_config()
    kb = KnowledgeBase(cfg)
    n = 0
    for meta in list_folder(service, folder_id):
        text = fetch_file_text(service, meta)
        if not text:
            continue
        kb.add_page(
            title=meta["name"],
            body=f"_Drive file: {meta['name']} (id={meta['id']})_\n\n{text}",
            category=category,
            tags=(tags or []) + ["drive"],
        )
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("rke.ingestion.drive")
    parser.add_argument("--folder", default=None, help="Drive folder ID (else read from sources.yaml)")
    parser.add_argument("--category", default="drive")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.folder:
        n = ingest_drive_folder(args.folder, category=args.category)
        print(f"  drive: {n} pages indexed from folder {args.folder}")
        return 0
    sources = load_sources().get("sources", {})
    drive_sources = {
        n: s for n, s in sources.items() if (s.get("type") or "") == "drive"
    }
    if not drive_sources:
        print("No drive: sources defined in config/sources.yaml; pass --folder.")
        return 1
    for name, spec in drive_sources.items():
        fid = spec.get("folder_id")
        if not fid:
            continue
        n = ingest_drive_folder(fid, category=spec.get("category", f"drive/{name}"), tags=spec.get("tags"))
        print(f"  {name}: {n} pages indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
