from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np
import requests
import urllib3

from garbage_vision.config import AppConfig

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _with_credentials(url: str, username: str, password: str) -> str:
    if not username or not password or "://" not in url or "@" in url:
        return url

    scheme, rest = url.split("://", 1)
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return f"{scheme}://{user}:{pwd}@{rest}"


def _with_reolink_query_credentials(url: str, username: str, password: str) -> str:
    if not username or not password:
        return url

    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("user", username)
    query.setdefault("password", password)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


def _with_query_values(url: str, values: dict[str, str]) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(values)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in {"password", "pass", "token", "key"}:
            query.append((key, "<redacted>"))
        else:
            query.append((key, value))
    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"<redacted>@{netloc.rsplit('@', 1)[1]}"
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), parts.fragment))


def _reolink_api_url(snapshot_url: str) -> str:
    parts = urlsplit(snapshot_url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _extract_reolink_error(payload: object) -> str:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            error = first.get("error")
            if isinstance(error, dict):
                detail = error.get("detail", "unknown error")
                rsp_code = error.get("rspCode", "unknown")
                warning = error.get("auth_warning_info")
                if isinstance(warning, dict) and "remain_times" in warning:
                    return f"{detail} rspCode={rsp_code} remain_times={warning['remain_times']}"
                return f"{detail} rspCode={rsp_code}"
    return str(payload)


def _reolink_login(config: AppConfig) -> str:
    if not config.camera_username or not config.camera_password:
        raise RuntimeError("CAMERA_USERNAME and CAMERA_PASSWORD are required for Reolink token login")

    login_url = _with_query_values(_reolink_api_url(config.camera_snapshot_url), {"cmd": "Login"})
    payload = [
        {
            "cmd": "Login",
            "param": {
                "User": {
                    "userName": config.camera_username,
                    "password": config.camera_password,
                }
            },
        }
    ]

    try:
        response = requests.post(
            login_url,
            json=payload,
            timeout=15,
            verify=config.camera_verify_tls,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Reolink login request failed for {_redact_url(login_url)}: {exc.__class__.__name__}"
        ) from None
    except ValueError:
        raise RuntimeError("Reolink login response was not valid JSON") from None

    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unexpected Reolink login response: {data}")

    first = data[0]
    if not isinstance(first, dict) or first.get("code") != 0:
        raise RuntimeError(f"Reolink login failed: {_extract_reolink_error(data)}")

    value = first.get("value")
    if not isinstance(value, dict) or not value.get("Token"):
        raise RuntimeError(f"Reolink login response did not include a token: {data}")
    return str(value["Token"].get("name", ""))


def _reolink_logout(config: AppConfig, token: str) -> None:
    if not token:
        return

    logout_url = _with_query_values(
        _reolink_api_url(config.camera_snapshot_url),
        {"cmd": "Logout", "token": token},
    )
    try:
        response = requests.post(
            logout_url,
            json=[{"cmd": "Logout"}],
            timeout=10,
            verify=config.camera_verify_tls,
        )
        response.raise_for_status()
    except requests.RequestException:
        LOGGER.warning("Reolink logout failed for %s", _redact_url(logout_url))


def read_snapshot(config: AppConfig) -> np.ndarray:
    url = config.camera_snapshot_url
    if not url:
        raise ValueError("CAMERA_SNAPSHOT_URL is required for snapshot mode")

    if not config.camera_verify_tls:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        token = _reolink_login(config)
        if not token:
            raise RuntimeError("Reolink login returned an empty token")
        url = _with_query_values(config.camera_snapshot_url, {"token": token})
        response = requests.get(url, timeout=15, verify=config.camera_verify_tls)
        response.raise_for_status()
    except requests.HTTPError as exc:
        redacted_url = _redact_url(url)
        content_type = response.headers.get("content-type", "unknown")
        raise RuntimeError(
            "Snapshot request failed for "
            f"{redacted_url}: HTTP {response.status_code} content-type={content_type}"
        ) from None
    except requests.RequestException as exc:
        redacted_url = _redact_url(url)
        raise RuntimeError(f"Snapshot request failed for {redacted_url}: {exc.__class__.__name__}") from None
    finally:
        if "token" in locals():
            _reolink_logout(config, token)

    image_bytes = np.frombuffer(response.content, dtype=np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if payload:
            raise RuntimeError(f"Snapshot response was not an image: {_extract_reolink_error(payload)}")
        raise RuntimeError("Snapshot response was not a readable image")
    return frame


def image_frames(image_dir: Path) -> Iterator[tuple[str, np.ndarray]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        LOGGER.warning("No test images found in %s", image_dir)

    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            LOGGER.warning("Skipping unreadable image: %s", path)
            continue
        yield path.name, frame


def video_frames(video_file: Path, every_n_frames: int = 30) -> Iterator[tuple[str, np.ndarray]]:
    if not video_file.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_file}")

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_file}")

    try:
        index = 0
        emitted = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every_n_frames == 0:
                yield f"{video_file.name}:frame-{index}", frame
                emitted += 1
            index += 1
        LOGGER.info("Read %s sampled frames from %s", emitted, video_file)
    finally:
        capture.release()


def rtsp_frame(config: AppConfig) -> np.ndarray:
    url = _with_credentials(config.camera_rtsp_url, config.camera_username, config.camera_password)
    if not url:
        raise ValueError("CAMERA_RTSP_URL is required for rtsp mode")

    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        raise RuntimeError("Could not open RTSP stream")

    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from RTSP stream")
        return frame
    finally:
        capture.release()


def camera_frame(config: AppConfig) -> np.ndarray:
    if config.camera_source == "snapshot":
        return read_snapshot(config)
    if config.camera_source == "rtsp":
        return rtsp_frame(config)
    raise ValueError("CAMERA_SOURCE must be either 'snapshot' or 'rtsp'")
