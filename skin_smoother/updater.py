"""
Keeps the desktop .exe up to date from the project's GitHub releases.

Each time the .exe starts, it asks GitHub once for the latest release. If that's
newer, the new .exe is downloaded next to the running one and checked: its size, that
it's a Windows program, and its SHA-256 against both GitHub's own record and the
checksum file published with it. Then it's swapped in. Windows lets a running
program be renamed but not overwritten, so the running copy becomes
"SkinSmoother.exe.old" and the new one takes its name. The next start uses the new
one and deletes the old copy.

The checksums catch a damaged or incomplete download. They come from the same place
as the .exe, so they don't protect against someone who controls the GitHub account.

Only a built .exe updates itself; running from source never does. Set
SKINSMOOTHER_NO_UPDATE=1 to turn it off. Nothing is sent but an ordinary request
for the release list: no photos, no personal details.
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from version import VERSION

REPO = "16sakuraa/PortraitColorLab"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
DOWNLOAD_PREFIX = f"https://github.com/{REPO}/releases/download/"
TEST_PREFIX = "http://127.0.0.1:"   # SKINSMOOTHER_UPDATE_URL may point here, for testing only
EXE_ASSET = "SkinSmoother.exe"
SUM_ASSET = EXE_ASSET + ".sha256"
TIMEOUT = 20          # seconds without an answer before giving up
RETRY_FOR = 10.0      # seconds to keep retrying a rename, e.g. while an antivirus scans the new file

_lock = threading.Lock()
_status = {"version": VERSION, "state": "off", "latest": None, "message": "", "reason": ""}


class UpdateError(Exception):
    """Why an update was abandoned, in words for the person using the program."""


def status() -> dict:
    """{version, state, latest, message, reason}. state: off, checking, current, offline,
    downloading, ready or failed. message is for showing, and empty when there's
    nothing to say; reason is why a failed update failed (the page words the rest
    itself, in its own language)."""
    with _lock:
        return dict(_status)


def _set(**changes) -> None:
    with _lock:
        _status.update(changes)
    if changes.get("message"):
        try:
            print("  Update:", changes["message"], flush=True)
        except (OSError, ValueError):         # no console, or one that can't show the text
            pass


def parse_version(text) -> tuple | None:
    """(major, minor, patch) from "v1.2.3" or "1.2.3"; None for anything else,
    including pre-releases such as "v1.2.3-rc1"."""
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", str(text or "").strip())
    return tuple(int(g) for g in m.groups()) if m else None


def _request(url: str):
    accept = "application/vnd.github+json" if url == LATEST_URL else "*/*"
    req = urllib.request.Request(url, headers={"User-Agent": f"SkinSmoother/{VERSION}", "Accept": accept})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _allowed(url: str, test: bool) -> bool:
    return isinstance(url, str) and url.startswith(TEST_PREFIX if test else DOWNLOAD_PREFIX)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _retry(action):
    """Run action, retrying for a while if Windows says the file is busy."""
    end = time.monotonic() + RETRY_FOR
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() > end:
                raise
            time.sleep(0.5)


def _expected_sums(exe_asset: dict, sum_asset: dict | None, test: bool) -> set:
    """Every SHA-256 the download must match: GitHub's record and the checksum file."""
    sums = set()
    digest = str(exe_asset.get("digest") or "")
    if digest.startswith("sha256:"):
        sums.add(digest[7:].lower())
    if sum_asset is not None:
        url = sum_asset.get("browser_download_url")
        if not _allowed(url, test):
            raise UpdateError("its checksum isn't on the project's GitHub page")
        with _request(url) as r:
            words = r.read(4096).decode("ascii", "replace").split()
        if not words or not re.fullmatch(r"[0-9a-fA-F]{64}", words[0]):
            raise UpdateError("its checksum file can't be read")
        sums.add(words[0].lower())
    if not sums:
        raise UpdateError("no checksum was published with it")
    if len(sums) > 1:
        raise UpdateError("its checksums don't agree")
    return sums


def _download(asset: dict, dest: str, sums: set, test: bool) -> None:
    url = asset.get("browser_download_url")
    if not _allowed(url, test):
        raise UpdateError("its download isn't on the project's GitHub page")
    size = int(asset.get("size") or 0)
    h, got = hashlib.sha256(), 0
    try:
        with _request(url) as r, open(dest, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                if got == 0 and not chunk.startswith(b"MZ"):
                    raise UpdateError("the download isn't a Windows program")
                got += len(chunk)
                if size and got > size:
                    raise UpdateError("the download was larger than expected")
                h.update(chunk)
                fh.write(chunk)
        if got == 0 or (size and got != size):
            raise UpdateError("the download stopped part-way")
        if h.hexdigest() not in sums:
            raise UpdateError("the download doesn't match its checksum")
    except BaseException:
        _remove(dest)
        raise


def _swap(exe: str, new: str) -> None:
    """Put new in exe's place, keeping the running copy as exe.old."""
    old = exe + ".old"
    if os.path.exists(old):
        try:
            _retry(lambda: os.remove(old))
        except OSError:
            raise UpdateError("an older copy of the program is still open") from None
    _retry(lambda: os.replace(exe, old))
    try:
        _retry(lambda: os.replace(new, exe))
    except OSError:
        _retry(lambda: os.replace(old, exe))     # put the running copy back
        raise


def run(exe: str, url: str = LATEST_URL) -> dict:
    """Check once, and install a newer release over exe. Never raises."""
    test = url.startswith(TEST_PREFIX)
    _set(state="checking", latest=None, message="", reason="")
    try:
        with _request(url) as r:
            release = json.loads(r.read(2_000_000).decode("utf-8"))
    except urllib.error.HTTPError as e:      # 404: nothing published yet
        _set(state="current" if e.code == 404 else "offline")
        return status()
    except (OSError, ValueError):            # offline, blocked, or a garbled answer
        _set(state="offline")
        return status()

    latest, mine = parse_version(release.get("tag_name")), parse_version(VERSION)
    if (latest is None or mine is None or latest <= mine
            or release.get("prerelease") or release.get("draft")):
        _set(state="current")
        return status()
    shown = ".".join(map(str, latest))
    new = exe + ".download"
    try:
        assets = {a.get("name"): a for a in release.get("assets") or [] if isinstance(a, dict)}
        if EXE_ASSET not in assets:
            raise UpdateError("the release has no SkinSmoother.exe")
        sums = _expected_sums(assets[EXE_ASSET], assets.get(SUM_ASSET), test)
        _set(state="downloading", latest=shown, message=f"Downloading version {shown}...")
        _download(assets[EXE_ASSET], new, sums, test)
        _swap(exe, new)
    except UpdateError as e:
        reason = str(e)
    except PermissionError:
        reason = ("Windows didn't allow saving it in this folder. "
                  "Move SkinSmoother.exe to your Desktop or Documents")
    except Exception:                         # noqa: BLE001 - a lost connection, a full disk...
        reason = "the download stopped part-way"
    else:
        _set(state="ready", latest=shown,
             message=f"Version {shown} is ready. To use it, close the Skin Smoother console "
                     "window and open the program again.")
        return status()
    _remove(new)
    _set(state="failed", latest=shown, reason=reason,
         message=f"Couldn't update to version {shown}: {reason}. It will try again next time.")
    return status()


def cleanup(exe: str) -> None:
    """Delete what a previous update left behind. A copy still open stays until next time."""
    _remove(exe + ".download")
    _remove(exe + ".old")


def start() -> None:
    """Check for an update in the background, if this is a built .exe."""
    if not getattr(sys, "frozen", False) or os.environ.get("SKINSMOOTHER_NO_UPDATE"):
        return
    exe = sys.executable
    url = os.environ.get("SKINSMOOTHER_UPDATE_URL", "")
    if not url.startswith(TEST_PREFIX):
        url = LATEST_URL
    cleanup(exe)
    _set(state="checking")
    threading.Thread(target=run, args=(exe, url), daemon=True).start()
