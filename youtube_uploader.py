r"""
╔══════════════════════════════════════════════════════════════════════════╗
║  YouTube Uploader  —  YouTubeSEO Sniper · ToonTalkStudios              ║
║  AI Thumbnail + SEO-optimized upload from seo_sniper JSON output       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Features                                                              ║
║  • Smart tag budget  — selects tags within YouTube's 500-char limit    ║
║  • Best title picker — auto-selects from 6 alternatives via score data ║
║  • AI thumbnail: Pollinations.ai FLUX (free) → NIM FLUX → Pillow      ║
║  • Pillow text overlay — headline + subtext from thumbnail_text_ideas  ║
║  • Chapter timestamps auto-injected for YouTube chapter detection      ║
║  • YouTube Data API v3 — OAuth cached, resumable, tqdm progress bar   ║
║  • Playwright fallback — browser automation when quota exceeded        ║
║  • Playlist auto-assignment by niche                                   ║
║  • Upload timing advisor from viral analysis                           ║
║  • Niche → YouTube category ID mapping                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
  # Full pipeline — upload + AI thumbnail:
  python youtube_uploader.py ./videos/ep01.mp4 ./output/ep01_seo.json

  # Generate + preview thumbnail only (no upload):
  python youtube_uploader.py ./videos/ep01.mp4 ./output/ep01_seo.json --thumbnail-only

  # Skip AI image — instant Pillow gradient thumbnail:
  python youtube_uploader.py ./videos/ep01.mp4 ./output/ep01_seo.json --no-ai-image

  # Pick a specific title (1=primary, 2-6=alternatives):
  python youtube_uploader.py ./videos/ep01.mp4 ./output/ep01_seo.json --use-title 3

  # Pre-made thumbnail (skip generation):
  python youtube_uploader.py ./videos/ep01.mp4 ./output/ep01_seo.json --thumbnail ./thumb.jpg

  # Browser-automation upload (no API quota used):
  python youtube_uploader.py ./videos/ep01.mp4 ./output/ep01_seo.json --playwright

  # Add to a playlist:
  python youtube_uploader.py ./videos/ep01.mp4 ./output/ep01_seo.json --playlist PLxxxxxxxxxx
"""

import os, sys, json, re, time, datetime, argparse, logging, textwrap, asyncio
from pathlib import Path
from typing import Optional
from io import BytesIO

import requests
from dotenv import load_dotenv

# ── PIL / Pillow ──────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

# ── tqdm progress bar ─────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── YouTube Data API ──────────────────────────────────────────────────────────
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    HAS_YTAPI = True
except ImportError:
    HAS_YTAPI = False

# ── Playwright ────────────────────────────────────────────────────────────────
try:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("yt_uploader")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

_cfg_path = Path(__file__).parent / "config.json"
CFG: dict = json.loads(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}

# ── YouTube constants (verified against YouTube Data API docs) ────────────────
# snippet.tags[]: combined length of all tag strings ≤ 500 chars.
# Tags with spaces are submitted as-is in the JSON array (API counts raw chars).
# In the Studio web UI the text input is also 500 chars including comma separators.
YT_TAG_LIMIT          = 500
YT_TITLE_LIMIT        = 100
YT_DESCRIPTION_LIMIT  = 5000
THUMBNAIL_W, THUMBNAIL_H = 1280, 720

# ── OAuth / credentials paths ─────────────────────────────────────────────────
CLIENT_SECRETS = Path(__file__).parent / "client_secrets.json"
TOKEN_FILE     = Path(__file__).parent / "yt_token.json"
PLAYWRIGHT_DIR = Path(__file__).parent / ".yt_pw_profile"

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

# ── NIM (re-uses existing project key) ───────────────────────────────────────
NIM_API_KEY  = os.getenv("NVIDIA_NIM_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── YouTube category ID map (niche substring → YT category) ──────────────────
_CATEGORY_MAP = {
    "dark psychology":   "26",   # Howto & Style
    "psychology":        "22",   # People & Blogs
    "relationships":     "22",
    "true crime":        "22",
    "self improvement":  "22",
    "spirituality":      "22",
    "philosophy":        "22",
    "history":           "27",   # Education
    "education":         "27",
    "ai":                "28",   # Science & Technology
    "technology":        "28",
    "gaming":            "20",   # Gaming
    "entertainment":     "24",   # Entertainment
    "animation":         "1",    # Film & Animation
    "health":            "26",
    "finance":           "22",
}

# ── Playlist assignment (fill in your playlist IDs in config.json) ────────────
PLAYLIST_MAP: dict = CFG.get("upload_playlist_map", {
    # "dark psychology": "PLxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SMART TAG OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def smart_tags(seo_data: dict, limit: int = YT_TAG_LIMIT) -> list:
    """
    Select the best tags within YouTube's 500-character combined limit.

    Priority order (highest ROI first):
      primary_tags → trending_tags → long_tail_tags → secondary_tags → competitor_gap_tags

    YouTube counts the sum of all individual tag string lengths.
    The API accepts tags as a JSON array (no comma separators counted).
    For the Playwright/web UI fallback a ", " separator IS counted — handled
    separately in the upload function via smart_tags_str().
    """
    buckets = [
        "primary_tags",
        "trending_tags",
        "long_tail_tags",
        "secondary_tags",
        "competitor_gap_tags",
    ]
    tags_sec = seo_data.get("tags", {})

    seen:    set  = set()
    ordered: list = []
    for key in buckets:
        for tag in tags_sec.get(key, []):
            clean = tag.strip()
            lower = clean.lower()
            if clean and lower not in seen:
                seen.add(lower)
                ordered.append(clean)

    selected: list = []
    used_chars = 0
    for tag in ordered:
        n = len(tag)
        if used_chars + n <= limit:
            selected.append(tag)
            used_chars += n
        if used_chars >= limit:
            break

    log.info(
        "🏷️   Tags: %d selected | %d / %d chars",
        len(selected), used_chars, limit,
    )
    return selected


def smart_tags_str(tags: list, limit: int = YT_TAG_LIMIT) -> str:
    """
    Build a comma-separated tag string within the character limit.
    Used by the Playwright uploader where the UI text input counts separators.
    """
    result = ""
    for tag in tags:
        candidate = f"{result}, {tag}" if result else tag
        if len(candidate) <= limit:
            result = candidate
        else:
            break
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BEST TITLE PICKER
# ══════════════════════════════════════════════════════════════════════════════

def best_title(seo_data: dict, viral_data: dict, force_index: int = 0) -> str:
    """
    Pick the title alternative that patches the video's weakest viral dimension.

    force_index = 0 → automatic selection (recommended)
    force_index = 1 → primary title
    force_index = 2-6 → alternatives[0-4]

    Auto-selection heuristic (based on viral score_breakdown):
      • trend_alignment  < 12 → trending-keyword-first variant (alts[4])
      • title_strength   < 13 → shock / controversy variant (alts[3])
      • search_demand    < 12 → curiosity-gap variant (alts[0])
      • emotional_impact < 13 → number / list variant (alts[1])
      • otherwise         → primary title
    """
    titles  = seo_data.get("titles", {})
    primary = titles.get("primary", "")
    alts    = titles.get("alternatives", [])
    scores  = viral_data.get("score_breakdown", {})

    all_t = [primary] + list(alts)

    if force_index > 0:
        idx = min(force_index - 1, len(all_t) - 1)
        chosen = all_t[idx]
        log.info("📝  Title (forced #%d): %s", force_index, chosen[:80])
        return chosen

    trend  = int(scores.get("trend_alignment",  15))
    title  = int(scores.get("title_strength",   15))
    search = int(scores.get("search_demand",    15))
    emot   = int(scores.get("emotional_impact", 15))

    if trend < 12 and len(alts) >= 5:
        chosen, reason = alts[4], f"trend_alignment={trend} — boosting trend relevance"
    elif title < 13 and len(alts) >= 4:
        chosen, reason = alts[3], f"title_strength={title} — boosting CTR with shock angle"
    elif search < 12 and len(alts) >= 1:
        chosen, reason = alts[0], f"search_demand={search} — boosting discoverability"
    elif emot < 13 and len(alts) >= 2:
        chosen, reason = alts[1], f"emotional_impact={emot} — using high-energy list format"
    else:
        chosen, reason = primary, "scores balanced — primary title wins"

    log.info("📝  Title auto-picked (%s): %s", reason, chosen[:80])
    return chosen[:YT_TITLE_LIMIT]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UPLOAD TIMING ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

def timing_advice(viral_data: dict) -> str:
    """
    Compare current UTC day/hour against the optimal upload window from the JSON.
    Returns a formatted advisory string. Does NOT block the upload.
    """
    now       = datetime.datetime.now(datetime.timezone.utc)
    today     = now.strftime("%A")
    hour      = now.hour

    bt        = viral_data.get("best_upload_times", {})
    best_days = bt.get("best_days",   [])
    avoid     = bt.get("avoid_days",  [])
    window    = bt.get("time_window", "")
    reason    = bt.get("reasoning",   "")

    lines = [f"⏰  Upload Timing Advisor  [{now.strftime('%A %H:%M')} UTC]"]

    if best_days:
        day_ok = today in best_days
        lines.append(
            f"  {'✅' if day_ok else '⚠️ '} Today ({today}) | Optimal: {', '.join(best_days)}"
            + (f" | Avoid: {', '.join(avoid)}" if avoid else "")
        )

    if window:
        m = re.search(r"(\d{1,2}):(\d{2})[–\-](\d{1,2}):(\d{2})", window)
        if m:
            h0, h1 = int(m.group(1)), int(m.group(3))
            in_window = h0 <= hour < h1
            lines.append(
                f"  {'✅' if in_window else '⚠️ '} Current: {now.strftime('%H:%M')} UTC"
                f" | Sweet spot: {window}"
            )

    if reason:
        lines.append(f"  💡 {reason}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THUMBNAIL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. FLUX prompt builder ───────────────────────────────────────────────────

def _build_flux_prompt(seo_data: dict, script_data: dict, viral_data: dict) -> str:
    """
    Build an optimised FLUX image-generation prompt from the SEO JSON.
    Targets YouTube thumbnail aesthetics: bold, dramatic, no text in image.
    """
    niche        = script_data.get("niche", "general")
    tone         = script_data.get("tone", "dark")
    thumb_ideas  = (
        viral_data.get("thumbnail_text_ideas") or
        seo_data.get("thumbnail_text_ideas") or []
    )
    viral_angles = viral_data.get("viral_angles", [{}])
    angle_concept= viral_angles[0].get("angle", "") if viral_angles else ""

    idea         = thumb_ideas[0] if thumb_ideas else {}
    color_scheme = idea.get("color_scheme", "dark red + white")
    style        = idea.get("style", "shock")

    # Map color schemes to descriptive language
    color_vocab = {
        "dark red + white":   "deep crimson tones, stark white highlights, deep shadows",
        "black + yellow":     "jet black background, saturated golden yellow, high contrast",
        "red + black":        "blood-red and pitch-black, ominous cinematic palette",
        "dark purple + gold": "deep violet atmosphere, gilded accents, mysterious luxury",
        "blue + white":       "cool deep blue, crisp white, modern clean aesthetic",
        "neon + dark":        "neon glow on dark background, cyberpunk vibes",
    }
    color_desc = next(
        (v for k, v in color_vocab.items() if k.lower() in color_scheme.lower()),
        "dark moody, high contrast, cinematic"
    )

    # Style → photographic direction
    style_vocab = {
        "shock":     "extreme close-up face with shocked expression, intense lighting, raw emotion",
        "curiosity": "silhouette partially revealed, fog, mysterious backlighting, intrigue",
        "warning":   "dramatic warning atmosphere, danger signals, urgent energy, red alert",
        "number":    "clean structural composition, geometric shapes, minimal balanced design",
    }
    style_desc = style_vocab.get(style, "dramatic cinematic composition, intense emotion")

    # Niche-specific visual language
    niche_vocab = {
        "dark psychology":   "lone silhouette in dramatic spotlight, shadow duality, psychological tension, Limbo aesthetic",
        "psychology":        "abstract mind visualization, introspective atmosphere, human figure contemplating",
        "relationships":     "two silhouettes in tension, emotional distance, dramatic backlighting",
        "true crime":        "dark urban environment, investigation atmosphere, cold case energy",
        "self improvement":  "figure breaking free from shadow, sunrise breakthrough, transformative energy",
        "ai":                "neural network visualization, glowing data streams, futuristic interface",
        "technology":        "sleek tech environment, blue-tinted ambient light, precision machinery",
        "history":           "ancient epic atmosphere, dramatic historical setting, time-worn textures",
        "animation":         "dynamic illustrated world, vibrant fantastical environment, rich color depth",
        "finance":           "power and wealth aesthetic, dramatic city skyline, financial tension",
    }
    niche_lower  = niche.lower()
    niche_visual = next(
        (v for k, v in niche_vocab.items() if k in niche_lower),
        "dramatic background, cinematic atmosphere, professional composition"
    )

    # Build final prompt — always end with no-text instructions
    parts = [
        niche_visual,
        style_desc,
        color_desc,
        "ultra-detailed, 8K, photorealistic, cinematic lighting, DSLR quality",
        "YouTube thumbnail background composition, rule of thirds",
    ]
    if angle_concept:
        parts.append(f"conceptual theme: {angle_concept.lower()}")

    # Critical negative instruction — must be at the end
    parts.append(
        "NO text, NO words, NO letters, NO subtitles, NO watermark, "
        "NO logo, NO typography, NO captions, NO numbers"
    )

    return ", ".join(parts)


# ── 4b. Image generation — Pollinations.ai (primary, free, no key) ──────────

def _generate_via_pollinations(prompt: str, w: int = 1280, h: int = 720) -> Optional[object]:
    """
    Pollinations.ai — COMPLETELY FREE, no API key, uses FLUX under the hood.
    Returns PIL Image or None on failure.
    """
    if not HAS_PILLOW:
        return None
    try:
        log.info("🌐  Pollinations.ai FLUX — generating thumbnail image...")
        r = requests.get(
            "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt),
            params={"width": w, "height": h, "model": "flux",
                    "seed": 42, "nologo": "true", "private": "true"},
            timeout=120,
        )
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize((w, h), Image.LANCZOS)
        log.info("✅  Pollinations image received (%dx%d)", img.width, img.height)
        return img
    except Exception as e:
        log.warning("Pollinations.ai failed: %s", e)
        return None


# ── 4c. Image generation — NIM FLUX (secondary, uses existing key) ──────────

def _generate_via_nim_flux(prompt: str, w: int = 1280, h: int = 704) -> Optional[object]:
    """
    NVIDIA NIM FLUX-schnell — uses project's existing API key.
    Height 704 (multiple of 32) → resized to 720 after.
    """
    if not HAS_PILLOW or not NIM_API_KEY:
        return None
    try:
        log.info("⚡  NIM FLUX-schnell — generating thumbnail image...")
        r = requests.post(
            f"{NIM_BASE_URL}/images/generations",
            headers={
                "Authorization": f"Bearer {NIM_API_KEY}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
            json={
                "model":           "black-forest-labs/flux-schnell",
                "prompt":          prompt,
                "n":               1,
                "width":           w,
                "height":          h,
                "response_format": "url",
            },
            timeout=120,
        )
        r.raise_for_status()
        data    = r.json()
        payload = data["data"][0]

        if payload.get("url", "").startswith("http"):
            img_bytes = requests.get(payload["url"], timeout=30).content
        else:
            import base64
            img_bytes = base64.b64decode(payload.get("b64_json", ""))

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize((1280, 720), Image.LANCZOS)
        log.info("✅  NIM FLUX image received")
        return img
    except Exception as e:
        log.warning("NIM FLUX failed: %s", e)
        return None


# ── 4d. Pure-Pillow gradient fallback ────────────────────────────────────────

def _pillow_gradient(w: int, h: int, color_scheme: str) -> Image.Image:
    """
    Fast gradient background using line-by-line paste (much faster than putpixel).
    Generates a dramatic two-tone vertical gradient matching the color scheme.
    """
    palettes = {
        "dark red + white":   ((160, 15, 15),  (8,  8,  8)),
        "black + yellow":     ((200, 160, 0),  (8,  8,  8)),
        "red + black":        ((180, 10, 10),  (4,  4,  4)),
        "dark purple + gold": ((70,  8, 110),  (8,  8,  8)),
        "blue + white":       ((15, 70, 160),  (8,  8,  8)),
        "neon + dark":        ((0,  200, 180), (4,  4, 20)),
    }
    cs = color_scheme.lower()
    top, bot = next((v for k, v in palettes.items() if k in cs), ((130, 15, 15), (8, 8, 8)))

    img = Image.new("RGB", (w, h))
    for y in range(h):
        t  = y / h
        r  = int(top[0] + (bot[0] - top[0]) * t)
        g  = int(top[1] + (bot[1] - top[1]) * t)
        b  = int(top[2] + (bot[2] - top[2]) * t)
        # paste a single-pixel-tall strip — faster than putpixel per pixel
        img.paste(Image.new("RGB", (w, 1), (r, g, b)), (0, y))

    # Subtle vignette: darken corners
    vignette = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vd       = ImageDraw.Draw(vignette)
    steps    = 40
    for i in range(steps):
        alpha = int(120 * (1 - i / steps) ** 2)
        pad   = i * 8
        vd.rectangle([pad, pad, w - pad, h - pad], outline=(0, 0, 0, alpha), width=8)

    return Image.alpha_composite(img.convert("RGBA"), vignette).convert("RGB")


# ── 4e. Font loader (cross-platform) ─────────────────────────────────────────

def _get_font(size: int) -> ImageFont:
    """Load the best available font for the platform."""
    candidates = []
    if sys.platform == "win32":
        candidates = [
            r"C:\Windows\Fonts\impact.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/Library/Fonts/Impact.ttf",
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    else:   # Linux / WSL
        candidates = [
            "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    log.warning("No TTF font found — using PIL default (upgrade: install msttcorefonts)")
    return ImageFont.load_default()


# ── 4f. Text overlay engine ───────────────────────────────────────────────────

def _overlay_text(
    base: Image.Image,
    thumb_ideas: list,
    seo_data: dict,
    viral_data: dict,
) -> Image.Image:
    """
    Composite text onto the base image.
    Layers (bottom to top):
      1. Base image
      2. Dark gradient bar (bottom 45%) for text readability
      3. Thin accent stripe at top (brand touch)
      4. HEADLINE — large Impact text with drop shadow
      5. Subtext — smaller text in accent color
    """
    if not HAS_PILLOW:
        return base

    idea         = thumb_ideas[0] if thumb_ideas else {}
    headline_raw = idea.get("headline") or ""
    subtext      = idea.get("subtext") or ""
    color_scheme = idea.get("color_scheme", "dark red + white")

    # Fallback headline from primary title (first 5 words, uppercased)
    if not headline_raw:
        primary = seo_data.get("titles", {}).get("primary", "")
        headline_raw = " ".join(primary.split()[:5]).upper()

    # ── Color palettes ────────────────────────────────────────────────────────
    palettes = {
        "dark red + white":   {"text": (255, 255, 255), "accent": (220, 30, 30),   "bar": (0, 0, 0)},
        "black + yellow":     {"text": (255, 220, 0),   "accent": (255, 255, 255), "bar": (5, 5, 5)},
        "red + black":        {"text": (255, 255, 255), "accent": (200, 20, 20),   "bar": (8, 8, 8)},
        "dark purple + gold": {"text": (255, 215, 0),   "accent": (200, 150, 255), "bar": (15, 0, 30)},
        "blue + white":       {"text": (255, 255, 255), "accent": (0,  150, 255),  "bar": (0, 15, 50)},
        "neon + dark":        {"text": (0,  255, 220),  "accent": (255, 80, 200),  "bar": (0, 0, 15)},
    }
    cs  = color_scheme.lower()
    pal = next((v for k, v in palettes.items() if k in cs), palettes["dark red + white"])

    w, h = base.size

    # ── Step 1: Dark gradient bar over bottom 45% ─────────────────────────────
    bar_h   = int(h * 0.45)
    bar_top = h - bar_h

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od      = ImageDraw.Draw(overlay)
    br, bg, bb = pal["bar"]
    for y in range(bar_top, h):
        alpha = int(210 * ((y - bar_top) / bar_h) ** 0.7)
        od.line([(0, y), (w, y)], fill=(br, bg, bb, alpha))

    result = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw   = ImageDraw.Draw(result)

    # ── Step 2: Accent stripe at very top (5px) ───────────────────────────────
    ar, ag, ab = pal["accent"]
    draw.rectangle([(0, 0), (w, 6)], fill=(ar, ag, ab))

    # ── Step 3: HEADLINE ──────────────────────────────────────────────────────
    MAX_FONT    = 110
    MIN_FONT    = 44
    MAX_WIDTH   = int(w * 0.93)
    MAX_LINES   = 3

    headline = headline_raw.upper()

    # Auto-wrap: try single line → 2 lines → 3 lines, shrink font until it fits
    best_font_size = MIN_FONT
    best_lines     = [headline]

    for wrap_w in [999, 18, 12]:
        wrapped = textwrap.fill(headline, width=wrap_w).split("\n")[:MAX_LINES]
        for fs in range(MAX_FONT, MIN_FONT - 1, -4):
            font = _get_font(fs)
            try:
                max_lw = max(draw.textlength(ln, font=font) for ln in wrapped)
            except AttributeError:
                max_lw = max(len(ln) * fs * 0.55 for ln in wrapped)
            if max_lw <= MAX_WIDTH:
                best_font_size = fs
                best_lines     = wrapped
                break

    font_h = _get_font(best_font_size)
    lh     = best_font_size + 8
    block_h = lh * len(best_lines)

    # Center text block in the bar area
    y_text = bar_top + max(16, (bar_h - block_h - 55) // 2)

    for i, line in enumerate(best_lines):
        try:
            lw = draw.textlength(line, font=font_h)
        except AttributeError:
            lw = len(line) * best_font_size * 0.55
        x = (w - lw) // 2
        y = y_text + i * lh

        # Multi-layer shadow for depth
        for dx, dy, alpha_frac in [(4, 4, 0.9), (2, 2, 0.6), (1, 1, 0.3)]:
            shadow_alpha = int(255 * alpha_frac)
            draw.text((x + dx, y + dy), line, font=font_h,
                      fill=(0, 0, 0, shadow_alpha) if hasattr(draw, "textlength") else (0, 0, 0))

        tr, tg, tb = pal["text"]
        draw.text((x, y), line, font=font_h, fill=(tr, tg, tb))

    # ── Step 4: Subtext ───────────────────────────────────────────────────────
    if subtext:
        sub_fs   = max(26, best_font_size // 3)
        font_sub = _get_font(sub_fs)
        sub_y    = y_text + block_h + 10

        for sub_line in textwrap.fill(subtext, width=55).split("\n"):
            try:
                sw = draw.textlength(sub_line, font=font_sub)
            except AttributeError:
                sw = len(sub_line) * sub_fs * 0.55
            sx = (w - sw) // 2
            draw.text((sx + 2, sub_y + 2), sub_line, font=font_sub, fill=(0, 0, 0))
            draw.text((sx, sub_y), sub_line, font=font_sub, fill=(ar, ag, ab))
            sub_y += sub_fs + 5

    return result


# ── 4g. Main thumbnail orchestrator ──────────────────────────────────────────

def generate_thumbnail(
    seo_data:    dict,
    script_data: dict,
    viral_data:  dict,
    output_path: str,
    use_ai:      bool = True,
) -> str:
    """
    Full thumbnail generation pipeline.
    Returns the path of the saved JPEG.

    Waterfall:
      [AI on]  Pollinations.ai FLUX (free, no key)
               → NIM FLUX (existing key)
               → Pillow gradient (instant, always works)
      [AI off] Pillow gradient only
    """
    if not HAS_PILLOW:
        log.error("Pillow not installed — pip install Pillow")
        return ""

    thumb_ideas  = (
        viral_data.get("thumbnail_text_ideas") or
        seo_data.get("thumbnail_text_ideas") or []
    )
    color_scheme = (thumb_ideas[0] if thumb_ideas else {}).get(
        "color_scheme", "dark red + white"
    )

    base: Optional[Image.Image] = None

    if use_ai:
        prompt = _build_flux_prompt(seo_data, script_data, viral_data)
        log.info("🖼️   FLUX prompt: %s", prompt[:100] + "…")
        base = _generate_via_pollinations(prompt)
        if base is None and NIM_API_KEY:
            base = _generate_via_nim_flux(prompt)

    if base is None:
        log.info("🎨  Using Pillow gradient (AI image unavailable or --no-ai-image)")
        base = _pillow_gradient(THUMBNAIL_W, THUMBNAIL_H, color_scheme)

    final = _overlay_text(base, thumb_ideas, seo_data, viral_data)

    out   = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save(str(out), "JPEG", quality=96, optimize=True, subsampling=0)

    size_kb = out.stat().st_size // 1024
    log.info("💾  Thumbnail → %s  [%dx%d | %dKB]",
             out.name, final.width, final.height, size_kb)
    return str(out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DESCRIPTION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_description(seo_data: dict, script_data: dict) -> str:
    """
    Assemble the final YouTube description from JSON output.
    Ensures timestamp block is present (triggers YouTube chapter detection).
    YouTube chapters need: ≥3 timestamps, first one must be 0:00.
    """
    desc_sec   = seo_data.get("description", {})
    full_text  = desc_sec.get("full_text",      "")
    timestamps = desc_sec.get("timestamps",      [])
    hashtags   = desc_sec.get("hashtags",        [])
    cta        = desc_sec.get("cta_elements",    [])

    parts = []
    if full_text:
        parts.append(full_text.strip())

    # Inject timestamp block if not already embedded
    if timestamps and not any(
        re.search(r"\d+:\d+", line) for line in (full_text or "").split("\n")[:30]
    ):
        parts.append("\n─────────\n" + "\n".join(timestamps))

    # CTA block
    if cta:
        parts.append("\n" + "\n".join(f"👉 {c}" for c in cta[:3]))

    # Hashtags at very end (YouTube shows first 3 under the title)
    if hashtags:
        tag_line = " ".join(h if h.startswith("#") else f"#{h}" for h in hashtags[:10])
        if tag_line not in full_text:
            parts.append("\n" + tag_line)

    return "\n\n".join(p for p in parts if p).strip()[:YT_DESCRIPTION_LIMIT]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — YOUTUBE OAUTH
# ══════════════════════════════════════════════════════════════════════════════

def get_youtube_client():
    """
    OAuth 2.0 — authenticate once, cache token forever.
    First run: browser opens for authorization.
    Subsequent runs: loads + auto-refreshes saved token.

    Setup (one-time):
      1. Google Cloud Console → APIs & Services → Enable YouTube Data API v3
      2. Create Credentials → OAuth 2.0 Client ID → Desktop App
      3. Download JSON → rename to client_secrets.json → place next to this script
    """
    if not HAS_YTAPI:
        raise RuntimeError(
            "YouTube API libs not installed.\n"
            "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )

    if not CLIENT_SECRETS.exists():
        raise FileNotFoundError(
            f"\nclient_secrets.json not found at:\n  {CLIENT_SECRETS}\n\n"
            "Get it from: Google Cloud Console → APIs & Services → Credentials\n"
            "→ Create Credentials → OAuth 2.0 Client ID → Desktop App → Download JSON\n"
            "→ Rename to client_secrets.json and place next to youtube_uploader.py\n"
            "→ Also: enable YouTube Data API v3 in your Google Cloud project.\n"
        )

    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), YOUTUBE_SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("🔄  Refreshing OAuth token...")
            creds.refresh(Request())
        else:
            log.info("🔐  Opening browser for YouTube authorization (one-time)...")
            flow  = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())
        log.info("💾  OAuth token saved → %s", TOKEN_FILE.name)

    return build("youtube", "v3", credentials=creds)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — YOUTUBE DATA API UPLOADER
# ══════════════════════════════════════════════════════════════════════════════

def _niche_to_category(niche: str) -> str:
    niche_l = niche.lower()
    for k, v in _CATEGORY_MAP.items():
        if k in niche_l:
            return v
    return "22"   # default: People & Blogs


def _playlist_id_for_niche(niche: str) -> Optional[str]:
    niche_l = niche.lower()
    for k, v in PLAYLIST_MAP.items():
        if k in niche_l or niche_l in k:
            return v
    return None


def upload_via_api(
    video_path:      str,
    seo_data:        dict,
    script_data:     dict,
    viral_data:      dict,
    title_override:  str = "",
    thumbnail_path:  str = "",
    playlist_id:     str = "",
    privacy:         str = "public",
) -> dict:
    """
    Upload via YouTube Data API v3.
    • Resumable multipart upload (4 MB chunks)
    • Real-time tqdm progress bar
    • Thumbnail set immediately after upload
    • Optional playlist assignment
    Returns: {"video_id": "...", "url": "..."}
    """
    youtube = get_youtube_client()

    title       = title_override or best_title(seo_data, viral_data)
    tags        = smart_tags(seo_data)
    description = build_description(seo_data, script_data)
    niche       = script_data.get("niche", "")
    category_id = _niche_to_category(niche)

    body = {
        "snippet": {
            "title":       title[:YT_TITLE_LIMIT],
            "description": description,
            "tags":        tags,
            "categoryId":  category_id,
        },
        "status": {
            "privacyStatus":           privacy,
            "selfDeclaredMadeForKids": False,
            "madeForKids":             False,
        },
    }

    log.info("▶️   Title      : %s", title[:80])
    log.info("🏷️   Tags       : %d tags | %d chars", len(tags), sum(len(t) for t in tags))
    log.info("📂  Category   : %s (ID %s)", niche or "default", category_id)
    log.info("🔒  Privacy    : %s", privacy)

    video_size_mb = Path(video_path).stat().st_size / 1024 / 1024
    log.info("⬆️   Uploading  : %s (%.1f MB)...", Path(video_path).name, video_size_mb)

    media   = MediaFileUpload(
        video_path,
        chunksize=4 * 1024 * 1024,   # 4 MB chunks — good balance for large files
        resumable=True,
        mimetype="video/*",
    )
    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    response  = None
    last_pct  = 0
    pbar      = tqdm(total=100, desc="  Upload", unit="%", ncols=64) if HAS_TQDM else None

    while response is None:
        status, response = request.next_chunk()
        if status and pbar:
            pct = int(status.progress() * 100)
            pbar.update(pct - last_pct)
            last_pct = pct

    if pbar:
        pbar.update(100 - last_pct)
        pbar.close()

    video_id = response.get("id", "")
    log.info("✅  Upload complete! video_id=%s  →  https://youtu.be/%s", video_id, video_id)

    # ── Set custom thumbnail ──────────────────────────────────────────────────
    if thumbnail_path and Path(thumbnail_path).exists() and video_id:
        try:
            log.info("🖼️   Setting thumbnail...")
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
            log.info("✅  Thumbnail set!")
        except Exception as e:
            log.warning("Thumbnail upload failed: %s", e)

    # ── Playlist assignment ───────────────────────────────────────────────────
    pid = playlist_id or _playlist_id_for_niche(niche)
    if pid and video_id:
        try:
            youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": pid,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            ).execute()
            log.info("📋  Added to playlist: %s", pid)
        except Exception as e:
            log.warning("Playlist assignment failed: %s", e)

    return {
        "video_id": video_id,
        "url":      f"https://youtu.be/{video_id}",
        "method":   "api",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PLAYWRIGHT FALLBACK UPLOADER
# ══════════════════════════════════════════════════════════════════════════════
#
# When to use:
#   • API quota exceeded (>~6 videos/day on free tier)
#   • OAuth client not yet verified
#   • Force with --playwright flag
#
# Behaviour:
#   • Runs headed (you can see & intervene)
#   • Saves a persistent browser profile so you stay logged in
#   • Uses resilient multi-selector fallback strategy
#
# NOTE: YouTube Studio DOM changes periodically. Selectors in this file
# are current as of June 2025. If selectors break, re-inspect Studio
# elements and update the _SELECTORS dict below.

_SEL = {
    # YouTube Studio DOM selectors (June 2025)
    "upload_btn":    'ytcp-icon-button#upload-icon, #upload-icon, [test-id="upload-icon"]',
    "upload_videos": 'tp-yt-paper-item:has-text("Upload videos"), yt-menu-renderer tp-yt-paper-item',
    "file_input":    'input#content, input[type=file][accept*=video]',
    "title_box":     '#title-container #textbox, ytcp-social-suggestions-textbox[id*=title] #textbox',
    "desc_box":      '#description-container #textbox, ytcp-social-suggestions-textbox[id*=description] #textbox',
    "thumb_input":   'input.label-file-input[accept*=image], input[type=file][accept*=image]',
    "show_more":     'ytcp-button#toggle-button[aria-label*="more" i], #toggle-button',
    "tags_input":    '#tags-container input, ytcp-chip-bar input',
    "next_btn":      'ytcp-button#next-button',
    "visibility":    'tp-yt-paper-radio-button[name="PUBLIC"]',
    "publish_btn":   'ytcp-button#done-button',
    "confirm_url":   'a[href*="youtu.be"], a[href*="youtube.com/watch"]',
}


def upload_via_playwright(
    video_path:     str,
    seo_data:       dict,
    script_data:    dict,
    viral_data:     dict,
    title_override: str = "",
    thumbnail_path: str = "",
) -> dict:
    """Run browser-automation YouTube Studio upload (headed Chromium)."""
    if not HAS_PLAYWRIGHT:
        raise RuntimeError(
            "Playwright not installed.\n"
            "Run: pip install playwright && playwright install chromium"
        )
    return asyncio.run(
        _playwright_upload(video_path, seo_data, script_data, viral_data,
                           title_override, thumbnail_path)
    )


async def _playwright_upload(
    video_path: str, seo_data: dict, script_data: dict, viral_data: dict,
    title_override: str, thumbnail_path: str,
) -> dict:
    """Async implementation of the Playwright Studio uploader."""
    title       = title_override or best_title(seo_data, viral_data)
    description = build_description(seo_data, script_data)
    tags_str    = smart_tags_str(smart_tags(seo_data), limit=480)  # ~480 for UI separators

    PLAYWRIGHT_DIR.mkdir(exist_ok=True)

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PLAYWRIGHT_DIR),
            headless=False,
            slow_mo=250,
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        # ── Navigate ──────────────────────────────────────────────────────────
        log.info("🌐  Opening YouTube Studio...")
        await page.goto("https://studio.youtube.com", timeout=30_000)
        await page.wait_for_load_state("networkidle", timeout=20_000)

        if "accounts.google.com" in page.url or "signin" in page.url:
            log.warning(
                "⚠️  Not logged in. Please sign in to your YouTube account in the "
                "browser window (you have 3 minutes)..."
            )
            await page.wait_for_url("**/studio.youtube.com**", timeout=180_000)

        # ── Click CREATE → Upload videos ──────────────────────────────────────
        log.info("🖱️   Click: Create button...")
        try:
            await page.locator(_SEL["upload_btn"]).first.click(timeout=10_000)
        except PWTimeout:
            # Some Studio versions use a button with text "Create"
            await page.get_by_role("button", name=re.compile("create", re.I)).first.click()

        await page.wait_for_timeout(800)

        try:
            await page.locator(_SEL["upload_videos"]).first.click(timeout=8_000)
        except PWTimeout:
            await page.get_by_text("Upload videos", exact=False).first.click()

        # ── Attach video file ─────────────────────────────────────────────────
        log.info("📂  Attaching video: %s", Path(video_path).name)
        try:
            async with page.expect_file_chooser(timeout=15_000) as fc_info:
                # Try clicking the visible "SELECT FILES" button
                try:
                    await page.get_by_role("button", name=re.compile("select file", re.I)).click(timeout=5_000)
                except PWTimeout:
                    await page.locator(_SEL["file_input"]).first.click(timeout=5_000)
            fc = await fc_info.value
            await fc.set_files(str(Path(video_path).resolve()))
        except Exception:
            # Direct set_input_files on hidden input
            await page.locator(_SEL["file_input"]).first.set_input_files(
                str(Path(video_path).resolve())
            )

        # Wait for the details panel to appear (title field loads)
        log.info("⌛  Waiting for upload details panel...")
        await page.wait_for_selector(_SEL["title_box"], timeout=60_000)
        await page.wait_for_timeout(1_500)

        # ── Fill TITLE ────────────────────────────────────────────────────────
        log.info("📝  Setting title: %s", title[:70])
        title_el = page.locator(_SEL["title_box"]).first
        await title_el.click()
        await page.keyboard.press("Control+A")
        await title_el.type(title[:YT_TITLE_LIMIT], delay=20)

        # ── Fill DESCRIPTION ─────────────────────────────────────────────────
        log.info("📝  Setting description (%d chars)...", len(description))
        desc_el = page.locator(_SEL["desc_box"]).first
        await desc_el.click()
        # fill() is faster than type() for long text
        await desc_el.fill(description)

        # ── Upload THUMBNAIL ──────────────────────────────────────────────────
        if thumbnail_path and Path(thumbnail_path).exists():
            log.info("🖼️   Uploading thumbnail...")
            try:
                thumb_el = page.locator(_SEL["thumb_input"]).first
                if await thumb_el.count() > 0:
                    await thumb_el.set_input_files(str(Path(thumbnail_path).resolve()))
                    await page.wait_for_timeout(2_000)
                    log.info("✅  Thumbnail attached")
            except Exception as e:
                log.warning("Thumbnail file input not found: %s", e)

        # ── SHOW MORE → TAGS ─────────────────────────────────────────────────
        log.info("🔽  Expanding 'Show more'...")
        try:
            await page.locator(_SEL["show_more"]).first.click(timeout=6_000)
            await page.wait_for_timeout(1_000)
        except PWTimeout:
            pass   # might already be expanded

        if tags_str:
            log.info("🏷️   Setting tags (%d chars)...", len(tags_str))
            try:
                tags_el = page.locator(_SEL["tags_input"]).first
                if await tags_el.count() > 0:
                    await tags_el.click()
                    await tags_el.fill(tags_str)
                    await tags_el.press("Enter")
                    await page.wait_for_timeout(500)
            except Exception as e:
                log.warning("Tags field not found: %s", e)

        # ── NEXT × 3 (Details → Video elements → Checks → Visibility) ────────
        for step in ("Video elements", "Checks", "Visibility"):
            log.info("➡️   Advancing to: %s", step)
            try:
                await page.locator(_SEL["next_btn"]).first.click(timeout=8_000)
            except PWTimeout:
                await page.get_by_role("button", name="Next").first.click()
            await page.wait_for_timeout(2_000)

        # ── Set PUBLIC ────────────────────────────────────────────────────────
        log.info("🔓  Setting visibility → Public...")
        try:
            await page.locator(_SEL["visibility"]).first.click(timeout=8_000)
        except PWTimeout:
            try:
                await page.get_by_label("Public").click(timeout=5_000)
            except PWTimeout:
                await page.get_by_text("Public", exact=True).first.click()

        await page.wait_for_timeout(800)

        # ── PUBLISH ───────────────────────────────────────────────────────────
        log.info("🚀  Publishing...")
        try:
            await page.locator(_SEL["publish_btn"]).first.click(timeout=8_000)
        except PWTimeout:
            await page.get_by_role("button", name=re.compile("publish|save", re.I)).first.click()

        # ── Wait for confirmation URL ─────────────────────────────────────────
        video_url = "Unknown (check YouTube Studio)"
        try:
            await page.wait_for_selector(_SEL["confirm_url"], timeout=90_000)
            link = await page.query_selector(_SEL["confirm_url"])
            if link:
                video_url = await link.get_attribute("href") or video_url
            log.info("✅  Published! → %s", video_url)
        except PWTimeout:
            log.warning("Could not detect confirmation URL — check YouTube Studio manually")

        await page.wait_for_timeout(3_000)
        await ctx.close()

    return {"url": video_url, "method": "playwright"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_upload_pipeline(
    video_path:       str,
    seo_json_path:    str,
    thumbnail_only:   bool = False,
    no_ai_image:      bool = False,
    use_title:        int  = 0,
    custom_thumbnail: str  = "",
    playlist_id:      str  = "",
    use_playwright:   bool = False,
    privacy:          str  = "public",
) -> dict:
    """
    Full upload pipeline:
      1. Load + validate JSON
      2. Print timing advice
      3. Generate thumbnail (Pollinations FLUX → NIM FLUX → Pillow gradient)
      4. Upload video (API → Playwright fallback)
      5. Return result dict
    """
    # ── Load JSON ─────────────────────────────────────────────────────────────
    seo_path = Path(seo_json_path)
    if not seo_path.exists():
        raise FileNotFoundError(f"SEO JSON not found: {seo_json_path}")

    full  = json.loads(seo_path.read_text(encoding="utf-8"))
    seo   = full.get("seo_optimization", {})
    scrpt = full.get("script_analysis",  {})
    viral = full.get("viral_analysis",   {})
    quick = full.get("quick_summary",    {})

    print(f"\n{'═'*62}")
    print(f"  🎯  YouTubeSEO Sniper — Upload Pipeline")
    print(f"  📄  SEO JSON    : {seo_path.name}")
    print(f"  🎬  Video       : {Path(video_path).name if not thumbnail_only else '(thumbnail only)'}")
    print(f"  🏆  Primary title : {quick.get('primary_title','N/A')[:60]}")
    print(f"  🔥  Viral score   : {quick.get('viral_score',0)}/100")
    print(f"  🏷️   Tag budget    : {sum(len(t) for t in smart_tags(seo))}/{YT_TAG_LIMIT} chars")
    print(f"{'═'*62}\n")

    print(timing_advice(viral))
    print()

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    thumb_path = custom_thumbnail
    if not thumb_path:
        stem      = seo_path.stem.replace("_seo", "")
        thumb_out = seo_path.parent / f"{stem}_thumbnail.jpg"
        thumb_path = generate_thumbnail(
            seo_data    = seo,
            script_data = scrpt,
            viral_data  = viral,
            output_path = str(thumb_out),
            use_ai      = not no_ai_image,
        )

    if thumbnail_only:
        print(f"\n✅  Thumbnail saved → {thumb_path}")
        if HAS_PILLOW and thumb_path:
            img = Image.open(thumb_path)
            print(f"    {img.width}×{img.height}px | {Path(thumb_path).stat().st_size//1024} KB")
        return {"thumbnail": thumb_path}

    # ── Upload ────────────────────────────────────────────────────────────────
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    title_chosen = best_title(seo, viral, force_index=use_title)

    if use_playwright:
        log.info("🎭  Mode: Playwright browser automation")
        result = upload_via_playwright(
            video_path     = video_path,
            seo_data       = seo,
            script_data    = scrpt,
            viral_data     = viral,
            title_override = title_chosen,
            thumbnail_path = thumb_path,
        )
    else:
        try:
            log.info("📡  Mode: YouTube Data API v3")
            result = upload_via_api(
                video_path     = video_path,
                seo_data       = seo,
                script_data    = scrpt,
                viral_data     = viral,
                title_override = title_chosen,
                thumbnail_path = thumb_path,
                playlist_id    = playlist_id,
                privacy        = privacy,
            )
        except Exception as e:
            log.warning("API upload failed: %s", e)
            if HAS_PLAYWRIGHT:
                log.info("🎭  Auto-falling back to Playwright...")
                result = upload_via_playwright(
                    video_path     = video_path,
                    seo_data       = seo,
                    script_data    = scrpt,
                    viral_data     = viral,
                    title_override = title_chosen,
                    thumbnail_path = thumb_path,
                )
            else:
                raise

    print(f"\n{'═'*62}")
    print(f"  ✅  Upload complete!")
    print(f"  🔗  Video URL  : {result.get('url','N/A')}")
    print(f"  🖼️   Thumbnail : {Path(thumb_path).name if thumb_path else 'none'}")
    print(f"  📡  Method    : {result.get('method','api')}")
    print(f"{'═'*62}\n")

    return {**result, "thumbnail": thumb_path}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="YouTubeSEO Sniper — Upload + AI Thumbnail Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    p.add_argument("video",            help="Video file path (.mp4, .mkv, etc.)")
    p.add_argument("seo_json",         help="seo_sniper output JSON path (*_seo.json)")
    p.add_argument("--thumbnail-only", action="store_true",
                   help="Generate & save thumbnail only — skip upload")
    p.add_argument("--no-ai-image",    action="store_true",
                   help="Skip Pollinations/NIM — use instant Pillow gradient thumbnail")
    p.add_argument("--use-title",      type=int, default=0, metavar="N",
                   help="Title: 1=primary, 2-6=alternatives (default: auto-select)")
    p.add_argument("--thumbnail",      default="",
                   help="Path to pre-made thumbnail (skips generation)")
    p.add_argument("--playlist",       default="",
                   help="YouTube playlist ID to add the video to")
    p.add_argument("--playwright",     action="store_true",
                   help="Use Playwright browser automation (no API quota used)")
    p.add_argument("--private",        action="store_true",
                   help="Upload as private instead of public")
    args = p.parse_args()

    try:
        run_upload_pipeline(
            video_path       = args.video,
            seo_json_path    = args.seo_json,
            thumbnail_only   = args.thumbnail_only,
            no_ai_image      = args.no_ai_image,
            use_title        = args.use_title,
            custom_thumbnail = args.thumbnail,
            playlist_id      = args.playlist,
            use_playwright   = args.playwright,
            privacy          = "private" if args.private else "public",
        )
    except KeyboardInterrupt:
        print("\n[CANCELLED]")
        sys.exit(1)
    except Exception as e:
        log.error("%s", e)
        raise


if __name__ == "__main__":
    main()
