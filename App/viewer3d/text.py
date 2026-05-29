"""Tiny PIL-based GL text overlay.

Each unique ``(text, color, font_px)`` triple is rasterized once by PIL into an
RGBA numpy array and uploaded as a ``GL_TEXTURE_2D``. Subsequent draws of the
same text just bind the cached texture, so the per-frame cost is one textured
quad – no flicker, no Tk widget on top of the GL canvas.

Designed to be called from inside the viewer's ``redraw()`` after setting an
orthographic projection.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_OK = True
except Exception:  # pragma: no cover - PIL is a hard dep elsewhere
    _PIL_OK = False

try:
    from OpenGL.GL import (
        GL_BLEND,
        GL_CLAMP_TO_EDGE,
        GL_LINEAR,
        GL_MODULATE,
        GL_NEAREST,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_QUADS,
        GL_REPLACE,
        GL_RGBA,
        GL_SRC_ALPHA,
        GL_TEXTURE_2D,
        GL_TEXTURE_ENV,
        GL_TEXTURE_ENV_MODE,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_UNPACK_ALIGNMENT,
        GL_UNSIGNED_BYTE,
        glBegin,
        glBindTexture,
        glBlendFunc,
        glColor4f,
        glDisable,
        glEnable,
        glEnd,
        glGenTextures,
        glPixelStorei,
        glTexCoord2f,
        glTexEnvi,
        glTexImage2D,
        glTexParameteri,
        glVertex2f,
    )

    _GL_OK = True
except Exception:  # pragma: no cover
    _GL_OK = False


def _find_default_font() -> str | None:
    """Return a usable TTF path, preferring high-quality system UI fonts."""
    candidates = []
    if sys.platform.startswith("win"):
        win = os.environ.get("WINDIR", r"C:\Windows")
        fonts_dir = os.path.join(win, "Fonts")
        for name in ("segoeui.ttf", "arial.ttf", "tahoma.ttf", "calibri.ttf", "consola.ttf"):
            candidates.append(os.path.join(fonts_dir, name))
    elif sys.platform == "darwin":
        candidates += [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


_DEFAULT_FONT_PATH = _find_default_font()


class TextRenderer:
    """LRU-ish cache of text → ``GL_TEXTURE_2D`` for 2D HUD drawing."""

    def __init__(self, font_size: int = 13, font_path: str | None = None) -> None:
        self.font_size = int(font_size)
        self.font_path = font_path if font_path else _DEFAULT_FONT_PATH
        self._font_cache: dict[int, "ImageFont.ImageFont"] = {}
        self._tex_cache: dict[tuple, tuple[int, int, int]] = {}

    def _get_font(self, px: int):
        if not _PIL_OK:
            return None
        if px in self._font_cache:
            return self._font_cache[px]
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, px)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        self._font_cache[px] = font
        return font

    def _make_texture(self, text: str, rgba: tuple[int, int, int, int], px: int) -> tuple[int, int, int]:
        font = self._get_font(px)
        if font is None or not _GL_OK:
            return 0, 0, 0

        # Use font metrics for canvas height: ascent + descent guarantees we
        # include descenders ('y','g','p','q') and the outline pixels. bbox
        # alone only spans rendered glyph pixels and can clip descenders.
        try:
            ascent, descent = font.getmetrics()
        except Exception:
            ascent, descent = px, max(2, px // 4)
        try:
            bbox = font.getbbox(text)
            text_w = max(1, bbox[2] - bbox[0])
            x_off = -int(bbox[0])  # shift if first glyph has negative bearing
        except Exception:
            text_w = max(1, int(px * len(text) * 0.6))
            x_off = 0

        # 3-px pad gives room for the 1-px black outline drawn in 8 directions
        # plus anti-aliasing tails without truncating the bottom of letters.
        pad = 3
        w = text_w + 2 * pad
        h = int(ascent) + int(descent) + 2 * pad
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            # Subtle 8-direction outline so labels are readable on any
            # background (mesh, sky, colorbar fill).
            outline = (0, 0, 0, 230)
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)):
                draw.text((pad + x_off + ox, pad + oy), text, fill=outline, font=font)
            draw.text((pad + x_off, pad), text, fill=rgba, font=font)
        except Exception:
            draw.text((pad, pad), text, fill=rgba)

        data = np.asarray(img, dtype=np.uint8)
        if data.ndim == 2:
            data = np.stack([data, data, data, data], axis=-1)
        elif data.shape[-1] == 3:
            alpha = np.full(data.shape[:2] + (1,), 255, dtype=np.uint8)
            data = np.concatenate([data, alpha], axis=-1)

        tid = glGenTextures(1)
        if isinstance(tid, (list, tuple, np.ndarray)):
            tex_id = int(np.asarray(tid).ravel()[0])
        else:
            tex_id = int(tid)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.tobytes())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        try:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        except Exception:
            pass
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id, w, h

    def get(self, text: str, rgba: tuple[int, int, int, int] = (235, 235, 235, 255), px: int | None = None) -> tuple[int, int, int]:
        px = int(px or self.font_size)
        key = (text, tuple(int(c) for c in rgba), px)
        if key in self._tex_cache:
            return self._tex_cache[key]
        try:
            entry = self._make_texture(text, key[1], px)
        except Exception:
            traceback.print_exc()
            entry = (0, 0, 0)
        self._tex_cache[key] = entry
        return entry

    def draw(
        self,
        text: str,
        x: float,
        y: float,
        rgba: tuple[int, int, int, int] = (235, 235, 235, 255),
        anchor: str = "nw",
        px: int | None = None,
    ) -> tuple[int, int]:
        """Draw text in the current 2D ortho projection. Returns ``(width, height)``."""
        if not _GL_OK:
            return 0, 0
        tex_id, w, h = self.get(text, rgba, px=px)
        if tex_id == 0:
            return 0, 0

        if anchor == "n":
            x0, y0 = x - w / 2, y
        elif anchor == "ne":
            x0, y0 = x - w, y
        elif anchor == "e":
            x0, y0 = x - w, y - h / 2
        elif anchor == "se":
            x0, y0 = x - w, y - h
        elif anchor == "s":
            x0, y0 = x - w / 2, y - h
        elif anchor == "sw":
            x0, y0 = x, y - h
        elif anchor == "w":
            x0, y0 = x, y - h / 2
        elif anchor == "center":
            x0, y0 = x - w / 2, y - h / 2
        else:  # "nw" default
            x0, y0 = x, y
        x1, y1 = x0 + w, y0 + h

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        try:
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        except Exception:
            pass
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x0, y0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x1, y0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x1, y1)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x0, y1)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        return w, h
