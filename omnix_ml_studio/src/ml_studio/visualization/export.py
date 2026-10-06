"""High-resolution figure export (no Streamlit dependency).

Raster formats (PNG, JPEG, WebP) are rendered at a chosen print width and DPI; vector
formats (SVG, PDF) are resolution independent. Static export uses Kaleido, which needs a
Chrome/Chromium binary: it is located automatically (``BROWSER_PATH``, the system PATH, or a
Playwright install). HTML export (interactive) never needs a browser.
"""
from __future__ import annotations

import glob
import io
import os
import re
import shutil
import threading
import zipfile

import plotly.graph_objects as go

FORMATS = {"PNG": "png", "JPEG": "jpeg", "WebP": "webp", "SVG": "svg", "PDF": "pdf", "HTML (interactive)": "html"}
MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "svg": "image/svg+xml",
        "pdf": "application/pdf", "html": "text/html"}
EXT = {"jpeg": "jpg"}
RASTER = {"png", "jpeg", "webp"}
BASE_WIDTH_PX = 1000          # layout width the figure is laid out at before scaling

_lock = threading.Lock()
_server = False


def find_browser() -> str | None:
    """Chrome / Chromium / Edge for Kaleido: BROWSER_PATH, the PATH, or the usual install folders
    on Windows, macOS and Linux (including a Playwright download)."""
    p = os.environ.get("BROWSER_PATH")
    if p and os.path.exists(p):
        return p
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome", "msedge"):
        w = shutil.which(name)
        if w:
            return w
    home = os.path.expanduser("~")
    pf = [os.environ.get(k, "") for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
    candidates = [os.path.join(b, "Google", "Chrome", "Application", "chrome.exe") for b in pf if b] + \
                 [os.path.join(b, "Microsoft", "Edge", "Application", "msedge.exe") for b in pf if b] + [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        os.path.join(home, "Applications", "Google Chrome.app", "Contents", "MacOS", "Google Chrome")]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    for pat in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                os.path.join(home, ".cache", "ms-playwright", "chromium-*", "chrome-linux", "chrome")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


_STATIC_OK = None


def static_available() -> bool:
    """True if PNG/SVG/PDF export works here (checked once with a tiny test figure)."""
    global _STATIC_OK
    if _STATIC_OK is not None:
        return _STATIC_OK
    try:
        import kaleido  # noqa: F401
    except ImportError:
        _STATIC_OK = False
        return False
    b = find_browser()
    if b:
        os.environ.setdefault("BROWSER_PATH", b)
    try:                                   # Kaleido may also find a browser on its own
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            go.Figure().to_image(format="png", width=10, height=10)
        _STATIC_OK = True
    except Exception:  # noqa: BLE001
        _STATIC_OK = False
    return _STATIC_OK


def slug(text: str, default: str = "figure") -> str:
    s = re.sub(r"<[^>]+>", " ", str(text or ""))
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return (s[:60] or default)


def title_of(fig: go.Figure) -> str:
    t = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    return t


def _prepare(fig: go.Figure, width_px: int) -> go.Figure:
    f = go.Figure(fig)
    h = f.layout.height or 450
    w = f.layout.width or width_px
    f.update_layout(width=w, height=h, paper_bgcolor="white", plot_bgcolor="white")
    return f


def scale_for(dpi: int, width_in: float, base_px: int = BASE_WIDTH_PX) -> float:
    return max(1.0, dpi * width_in / base_px)


def _set_png_dpi(data: bytes, dpi: int) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return data
    im = Image.open(io.BytesIO(data))
    out = io.BytesIO()
    im.save(out, format="PNG", dpi=(dpi, dpi))
    return out.getvalue()


def export(fig: go.Figure, fmt: str, dpi: int = 300, width_in: float = 7.0) -> bytes:
    """Bytes of ``fig`` in ``fmt`` (png, jpeg, webp, svg, pdf, html)."""
    fmt = FORMATS.get(fmt, fmt)
    if fmt == "html":
        return go.Figure(fig).to_html(include_plotlyjs=True, full_html=True,
                                      config={"displaylogo": False}).encode()
    if not static_available():
        raise RuntimeError("Static image export needs Chrome or Chromium on the server. Install it (Docker image "
                           "and packages.txt include it) or download the interactive HTML instead.")
    global _server
    f = _prepare(fig, BASE_WIDTH_PX)
    scale = scale_for(dpi, width_in) if fmt in RASTER else 1.0
    with _lock:
        if not _server:
            try:
                import kaleido
                kaleido.start_sync_server(silence_warnings=True)
                _server = True
            except Exception:  # noqa: BLE001 - fall back to per-call rendering
                _server = True
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = f.to_image(format=fmt, scale=scale)
    if fmt == "png":
        data = _set_png_dpi(data, dpi)
    return data


def filename(name: str, fmt: str) -> str:
    fmt = FORMATS.get(fmt, fmt)
    return f"{slug(name)}.{EXT.get(fmt, fmt)}"


def export_zip(figs: list[tuple[str, go.Figure]], fmts: list[str], dpi: int = 300, width_in: float = 7.0) -> bytes:
    out = io.BytesIO()
    seen = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, fig in figs:
            base = slug(name)
            seen[base] = seen.get(base, 0) + 1
            if seen[base] > 1:
                base = f"{base}_{seen[base]}"
            for fm in fmts:
                fm = FORMATS.get(fm, fm)
                z.writestr(f"{fm}/{base}.{EXT.get(fm, fm)}", export(fig, fm, dpi, width_in))
    return out.getvalue()
