"""Cross-platform matplotlib font helpers for CJK text rendering."""

from __future__ import annotations

import logging
from pathlib import Path

from matplotlib import ft2font, font_manager, pyplot as plt

LOGGER = logging.getLogger(__name__)

PREFERRED_FONT_FAMILIES = [
    "PingFang SC",
    "PingFang HK",
    "Songti SC",
    "Hiragino Sans GB",
    "STHeiti",
    "Heiti TC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "Heiti SC",
    "Sarasa Gothic SC",
    "DejaVu Sans",
]

PREFERRED_FONT_FILES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def _available_font_names() -> set[str]:
    names: set[str] = set()
    for font in font_manager.fontManager.ttflist:
        if font.name:
            names.add(font.name)
    return names


def _font_supports_sample_text(font_path: str, sample_text: str = "古典弱智奇怪提问") -> bool:
    try:
        font = ft2font.FT2Font(font_path)
        charmap = font.get_charmap()
        return all(ord(ch) in charmap for ch in sample_text)
    except Exception:
        return False


def _discover_cjk_fonts_from_matplotlib() -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()
    priority_keywords = (
        "pingfang",
        "hiragino",
        "heiti",
        "stheiti",
        "songti",
        "simsong",
        "arial unicode",
        "noto",
        "source han",
        "yahei",
        "simhei",
        "wenquanyi",
        "sarasa",
        "cjk",
    )

    for font in font_manager.fontManager.ttflist:
        font_name = font.name or ""
        font_path = getattr(font, "fname", "")
        lowered = font_name.lower()
        if not font_path or font_name in seen:
            continue
        if not any(keyword in lowered for keyword in priority_keywords):
            continue
        if _font_supports_sample_text(font_path):
            discovered.append(font_name)
            seen.add(font_name)

    return discovered


def _existing_font_files() -> list[str]:
    found: list[str] = []
    for raw_path in PREFERRED_FONT_FILES:
        path = Path(raw_path)
        if path.exists():
            found.append(str(path))
    return found


def configure_matplotlib_cjk_fonts() -> list[str]:
    """Configure matplotlib to use the first available CJK-capable fonts."""
    font_names = _available_font_names()
    selected_names = [name for name in PREFERRED_FONT_FAMILIES if name in font_names]
    for name in _discover_cjk_fonts_from_matplotlib():
        if name not in selected_names:
            selected_names.append(name)

    for font_file in _existing_font_files():
        try:
            font_manager.fontManager.addfont(font_file)
            font_name = font_manager.FontProperties(fname=font_file).get_name()
            if font_name and font_name not in selected_names:
                selected_names.append(font_name)
        except Exception as exc:
            LOGGER.debug("Failed to register font file %s: %s", font_file, exc)

    selected_names = list(dict.fromkeys(selected_names))

    if "DejaVu Sans" not in selected_names:
        selected_names.append("DejaVu Sans")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = selected_names
    if selected_names:
        plt.rcParams["font.family"] = [selected_names[0], "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected_names