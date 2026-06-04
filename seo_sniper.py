r"""
╔══════════════════════════════════════════════════════════════════════╗
║   YouTubeSEO Sniper v1.2  —  ToonTalkStudios / Nour Fawzy            ║
╠══════════════════════════════════════════════════════════════════════╣
║  NEW in v1.2 (fixes 2179s timeout run)                               ║
║  ✅ Non-streaming calls — NIM buffers; streaming was causing 300s   ║
║     per-chunk timeouts → 30+ min wasted on Agents 4 & 5              ║
║  ✅ Per-provider timeouts: NIM 180s | Groq 60s | OpenRouter 90s     ║
║  ✅ Cross-provider fallback: NIM timeout → Groq → OpenRouter        ║
║     Set GROQ_API_KEY in .env to unlock instant fallback              ║
║  ✅ Google Trends 429 — 3s sleep + tz change to reduce bot detect  ║
║  ✅ DDG rate limit — delay 0.8s → 2.5s between queries             ║
║  ✅ Reduced token budgets: 3→2500 | 4→2000 | 5→2000+1500          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, re, time, datetime, argparse, logging, hashlib
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv


# ── Compatibility: pytrends + urllib3 >= 2.0 ──────────────────────────
# pytrends creates urllib3.Retry objects using `method_whitelist=`, which
# was renamed to `allowed_methods=` in urllib3 2.0.  Patch it silently
# at import time so pytrends works without any version pinning.
try:
    from urllib3.util.retry import Retry as _Retry
    _orig_retry_init = _Retry.__init__
    def _patched_retry_init(self, *args, **kwargs):
        kwargs.pop("method_whitelist", None)   # drop deprecated kwarg
        _orig_retry_init(self, *args, **kwargs)
    _Retry.__init__ = _patched_retry_init
    del _Retry
except Exception:
    pass


# ── Optional dependencies ──────────────────────────────────────────────
try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False

try:                              # new package name (2024 rename)
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    try:                          # old name fallback
        from duckduckgo_search import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False


# ── Load env & config ─────────────────────────────────────────────────
load_dotenv()

_cfg_path = Path(__file__).parent / "config.json"
CFG: dict = json.loads(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}

CACHE_DIR     = Path(__file__).parent / ".seo_cache"
CACHE_TTL_MIN = int(CFG.get("cache_ttl_minutes", 60))


# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seo_sniper")


# ══════════════════════════════════════════════════════════════════════
# MULTI-PROVIDER LLM CLIENT
# ══════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    Unified streaming LLM client — NIM │ Groq │ OpenRouter.
    All three providers expose an OpenAI-compatible /v1/chat/completions
    endpoint with Server-Sent Events (SSE) streaming.
    Streaming means the TCP connection stays alive while tokens arrive,
    so large JSON responses never hit a read-timeout.
    """

    _PROVIDERS = {
        "nim": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "env_key":  "NVIDIA_NIM_API_KEY",
            "models": {
                "parse": "meta/llama-3.3-70b-instruct",
                "seo":   "mistralai/mistral-small-4-119b-2603",
                "viral": "mistralai/mistral-small-4-119b-2603",
                "ideas": "mistralai/mistral-small-4-119b-2603",
            },
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "env_key":  "GROQ_API_KEY",
            "models": {
                "parse": "llama-3.3-70b-versatile",
                "seo":   "llama-3.3-70b-versatile",
                "viral": "llama-3.3-70b-versatile",
                "ideas": "llama-3.3-70b-versatile",
            },
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "env_key":  "OPENROUTER_API_KEY",
            "models": {
                "parse": "meta-llama/llama-3.3-70b-instruct",
                "seo":   "qwen/qwen3-235b-a22b:free",
                "viral": "qwen/qwen3-235b-a22b:free",
                "ideas": "qwen/qwen3-235b-a22b:free",
            },
        },
    }

    def __init__(self, provider: str = "nim", override_cfg: dict = None):
        p = provider.lower()
        if p not in self._PROVIDERS:
            raise ValueError(
                f"Unknown provider '{p}'. Choose from: {list(self._PROVIDERS)}"
            )
        info = self._PROVIDERS[p]
        ov   = override_cfg or {}

        self.provider = p
        self.base_url = ov.get("base_url", info["base_url"])
        self.api_key  = os.getenv(ov.get("env_key", info["env_key"]), "")
        self.models   = {**info["models"], **ov.get("models", {})}

    def get_model(self, role: str) -> str:
        return self.models.get(role, self.models.get("seo", ""))

    # ── Per-provider read timeouts (seconds) ─────────────────────────────────
    # NIM free tier buffers responses rather than streaming them — the full
    # response only arrives once generation is complete.  180s gives the free
    # tier a fair window without waiting forever.  Groq / OpenRouter are much
    # faster so shorter timeouts are fine.
    _TIMEOUTS = {
        "nim":        (15, 180),
        "groq":       (10,  60),
        "openrouter": (10,  90),
    }

    def call(self, model: str, messages: list, max_tokens: int = 2048,
             temperature: float = 0.3, retries: int = 2) -> str:
        """
        Non-streaming POST to /v1/chat/completions.

        Why non-streaming:  NIM free tier buffers the full response before
        sending — streaming SSE gives no benefit and the per-chunk timeout
        fired prematurely (300s × 3 retries = 15 min of wasted time).

        On timeout the call is forwarded to `_try_fallbacks()` which
        transparently retries on Groq → OpenRouter if those keys are set.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        timeout = self._TIMEOUTS.get(self.provider, (15, 120))

        for attempt in range(1, retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers, json=payload,
                    timeout=timeout,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()

            except requests.exceptions.ConnectTimeout:
                log.warning("Connect timeout (attempt %d/%d)", attempt, retries)
                time.sleep(5)

            except requests.exceptions.ReadTimeout:
                log.warning(
                    "Read timeout after %ds (attempt %d/%d) — "
                    "NIM may be under load; trying fallback providers...",
                    timeout[1], attempt, retries,
                )
                # Don't burn more retries on the same slow provider — fall
                # through to cross-provider fallback immediately.
                return self._try_fallbacks(model, messages, max_tokens, temperature)

            except requests.HTTPError as e:
                status = getattr(r, "status_code", 0)
                log.warning("HTTP %d (attempt %d/%d): %s", status, attempt, retries, e)
                if status == 429:
                    time.sleep(min(60, 10 * attempt))
                elif status >= 500:
                    time.sleep(2 ** attempt)
                else:
                    break

            except Exception as e:
                log.warning("LLM error (attempt %d/%d): %s", attempt, retries, e)
                if attempt < retries:
                    time.sleep(2 ** attempt)

        return self._try_fallbacks(model, messages, max_tokens, temperature)

    def _try_fallbacks(self, model: str, messages: list,
                       max_tokens: int, temperature: float) -> str:
        """
        Transparently retry on fallback providers (Groq → OpenRouter).
        Only tries providers whose API key is present in the environment.
        Configured via  config.json → "fallback_providers": ["groq", "openrouter"]
        """
        fb_names = CFG.get("fallback_providers", [])
        # Map the current model string back to its role (parse / seo / viral / ideas)
        role = next((r for r, m in self.models.items() if m == model), "seo")

        for fb_name in fb_names:
            if fb_name == self.provider:
                continue
            fb_defaults = self._PROVIDERS.get(fb_name, {})
            fb_key      = os.getenv(fb_defaults.get("env_key", ""), "")
            if not fb_key:
                log.debug("Fallback '%s' skipped — no API key set", fb_name)
                continue

            fb_cfg   = CFG.get("providers", {}).get(fb_name, {})
            fb_url   = fb_cfg.get("base_url", fb_defaults["base_url"])
            fb_model = {**fb_defaults.get("models", {}),
                        **fb_cfg.get("models", {})}.get(role, fb_defaults["models"].get("seo", ""))
            fb_to    = self._TIMEOUTS.get(fb_name, (10, 60))

            log.info("🔄  Fallback → %s / %s", fb_name, fb_model.split("/")[-1])
            try:
                r = requests.post(
                    f"{fb_url}/chat/completions",
                    headers={"Authorization": f"Bearer {fb_key}",
                             "Content-Type":  "application/json"},
                    json={"model": fb_model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=fb_to,
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                if content:
                    log.info("✅  Fallback %s succeeded", fb_name)
                    return content
            except Exception as e:
                log.warning("Fallback %s failed: %s", fb_name, e)

        log.error("All providers exhausted — returning empty string")
        return ""

    def json_call(self, model: str, system: str, user: str,
                  max_tokens: int = 3000) -> dict:
        """Call LLM and return robustly-parsed JSON."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        raw = self.call(model, messages, max_tokens)
        return _repair_json(raw) if raw else {}


# ── Module-level client — init on import, optionally re-init via CLI ──
_provider_name = CFG.get("provider", "nim")
_provider_cfg  = CFG.get("providers", {}).get(_provider_name, {})
_llm: LLMClient = LLMClient(_provider_name, _provider_cfg)

# Backward-compat globals (used by pipeline_hook.py)
NIM_API_KEY  = _llm.api_key
MODEL_PARSE  = _llm.get_model("parse")
MODEL_SEO    = _llm.get_model("seo")
MODEL_VIRAL  = _llm.get_model("viral")
MODEL_IDEAS  = _llm.get_model("ideas")


def _reinit_client(provider: str) -> None:
    """Switch provider at CLI startup before any agents run."""
    global _llm, NIM_API_KEY, MODEL_PARSE, MODEL_SEO, MODEL_VIRAL, MODEL_IDEAS
    prov_cfg = CFG.get("providers", {}).get(provider, {})
    _llm       = LLMClient(provider, prov_cfg)
    NIM_API_KEY = _llm.api_key
    MODEL_PARSE = _llm.get_model("parse")
    MODEL_SEO   = _llm.get_model("seo")
    MODEL_VIRAL = _llm.get_model("viral")
    MODEL_IDEAS = _llm.get_model("ideas")


# Backward-compat alias for pipeline_hook.py
def nim_json(model: str, system: str, user: str, max_tokens: int = 3000) -> dict:
    return _llm.json_call(model, system, user, max_tokens)


# ══════════════════════════════════════════════════════════════════════
# 5-STRATEGY JSON REPAIR
# ══════════════════════════════════════════════════════════════════════

def _repair_json(raw: str) -> dict:
    """
    Robustly extract a JSON dict from an LLM response.
    Tries 5 strategies from least to most invasive.
    """
    s = raw.strip()

    # 1 — direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2 — strip markdown fences  (```json ... ```)
    clean = re.sub(r"```(?:json|JSON)?\s*|\s*```", "", s).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # 3 — extract the first complete { } block
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 4 — common syntax fixes
    fixed = clean
    fixed = re.sub(r",\s*([\}\]])", r"\1", fixed)  # trailing commas
    fixed = re.sub(r"\bTrue\b",  "true",  fixed)   # Python literals
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b",  "null",  fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5 — truncation recovery: count unclosed braces and close them
    if fixed.lstrip().startswith("{"):
        depth, in_str, esc = 0, False, False
        for ch in fixed:
            if esc:             esc = False; continue
            if ch == "\\":      esc = True;  continue
            if ch == '"':       in_str = not in_str; continue
            if not in_str:
                if ch in "{[":  depth += 1
                elif ch in "}]": depth -= 1
        if depth > 0:
            try:
                return json.loads(fixed + "}" * depth)
            except json.JSONDecodeError:
                pass

    log.error("JSON repair failed — all 5 strategies exhausted")
    log.debug("Raw (first 500): %s", raw[:500])
    return {}


# ══════════════════════════════════════════════════════════════════════
# TREND CACHE  (file-based, configurable TTL)
# ══════════════════════════════════════════════════════════════════════

def _cache_key(queries: list, niche: str) -> str:
    payload = json.dumps(sorted(q.lower() for q in queries) + [niche.lower()])
    return hashlib.md5(payload.encode()).hexdigest()


def _cache_read(key: str) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        data  = json.loads(p.read_text(encoding="utf-8"))
        ts    = data.get("collected_at", "")
        saved = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age   = (datetime.datetime.now(datetime.timezone.utc) - saved).seconds / 60
        if age < CACHE_TTL_MIN:
            log.info(f"📦  Trend cache hit — {age:.0f}m old (TTL={CACHE_TTL_MIN}m)")
            return data
        p.unlink()   # expired
    except Exception:
        pass
    return None


def _cache_write(key: str, data: dict) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        log.debug("Cache write failed: %s", e)


# ══════════════════════════════════════════════════════════════════════
# PROGRESSIVE SAVE  (write partial JSON after each agent)
# ══════════════════════════════════════════════════════════════════════

def _partial_save(out_path: str, payload: dict) -> None:
    """
    Write whatever we have so far.  Protects against later-agent failures:
    if Agent 5 crashes, you still have all the SEO & viral data on disk.
    """
    try:
        Path(out_path).write_text(
            json.dumps({**payload, "_partial": True}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        log.debug("Partial save error: %s", e)


# ══════════════════════════════════════════════════════════════════════
# OUTPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════

def _validate_output(output: dict) -> dict:
    """Enforce YouTube upload constraints and coerce types."""
    seo   = output.get("seo_optimization", {})
    viral = output.get("viral_analysis",   {})
    qs    = output.get("quick_summary",    {})
    cfg_s = CFG.get("seo_output", {})

    # Tags — enforce max_tags limit
    max_tags = int(cfg_s.get("max_tags", 40))
    tags     = seo.get("tags", {})
    all_tags = tags.get("all_tags", [])
    if len(all_tags) > max_tags:
        tags["all_tags"] = all_tags[:max_tags]

    # Primary title — enforce max_title_chars
    max_chars = int(cfg_s.get("max_title_chars", 70))
    titles    = seo.get("titles", {})
    primary   = titles.get("primary", "")
    if len(primary) > max_chars:
        trimmed    = primary[:max_chars]
        last_space = trimmed.rfind(" ")
        titles["primary"] = trimmed[:last_space] if last_space > 40 else trimmed
        log.warning("Title trimmed to %d chars: %s", max_chars, titles["primary"])

    # Viral score — clamp 0-100, coerce to int
    raw_score = viral.get("viral_score", 0)
    try:
        raw_score = int(float(str(raw_score)))
    except (TypeError, ValueError):
        raw_score = 0
    viral["viral_score"] = max(0, min(100, raw_score))

    # Keep quick_summary in sync with validated values
    qs["viral_score"]   = viral["viral_score"]
    qs["primary_title"] = titles.get("primary", qs.get("primary_title", "N/A"))

    return output


# ══════════════════════════════════════════════════════════════════════
# AGENT 1 — SCRIPT PARSER
# ══════════════════════════════════════════════════════════════════════

_PARSE_SYS = (
    "You are a professional YouTube content analyst. "
    "Extract structured metadata from video scripts. "
    "Respond with valid JSON only — no markdown, no preamble."
)


def agent_parse_script(script: str) -> dict:
    log.info("🔍  [Agent 1/5] Script Parser — analyzing content...")
    prompt = f"""Analyze this YouTube video script and extract ALL metadata fields below.

SCRIPT (first 6000 chars):
{script[:6000]}

Return EXACTLY this JSON (no extra keys, fill every field):
{{
  "main_topic": "short topic name",
  "content_summary": "2-3 sentence summary",
  "niche": "primary niche",
  "sub_niche": "specific sub-niche",
  "content_type": "educational|story|tutorial|commentary|review|entertainment",
  "tone": "dark|inspirational|informative|shocking|calm|humorous",
  "target_audience": "who this appeals to",
  "estimated_duration_mins": 8,
  "key_entities": ["people/brands/places mentioned"],
  "main_topics": ["topic1", "topic2", "topic3"],
  "subtopics": ["sub1", "sub2", "sub3", "sub4"],
  "core_message": "one sentence — what is the main takeaway",
  "emotional_triggers": ["curiosity", "fear", "trust", "urgency"],
  "unique_angle": "what makes this video different from others on the topic",
  "content_pillars": ["pillar1", "pillar2"],
  "seed_keywords": ["8 keywords that naturally appear or fit the script"],
  "related_searches": ["what viewers would search after watching — 6 phrases"],
  "competitor_topics": ["similar videos already on YouTube — 4 examples"],
  "difficulty_level": "beginner|intermediate|advanced"
}}"""
    result = _llm.json_call(MODEL_PARSE, _PARSE_SYS, prompt, max_tokens=1800)
    if not result:
        log.warning("Script parsing fallback — minimal metadata")
        result = {
            "main_topic":        Path().cwd().name,
            "niche":             "general",
            "content_type":      "educational",
            "seed_keywords":     [],
            "subtopics":         [],
            "emotional_triggers":[],
        }
    return result


# ══════════════════════════════════════════════════════════════════════
# AGENT 2 — TREND INTELLIGENCE  (4 sources, fully parallel)
# ══════════════════════════════════════════════════════════════════════

_SUBREDDITS = CFG.get("subreddit_map", {
    "dark psychology":   ["psychology", "manipulation", "socialengineering", "coercivecontrol", "NarcissisticAbuse"],
    "psychology":        ["psychology", "socialpsychology", "AskPsychology", "cognitivescience"],
    "true crime":        ["TrueCrime", "unresolvedmysteries", "criminalminds", "serialkillers"],
    "self improvement":  ["selfimprovement", "productivity", "getmotivated", "DecidingToBeBetter"],
    "ai":                ["artificial", "MachineLearning", "LocalLLaMA", "ChatGPT", "singularity"],
    "technology":        ["technology", "Futurology", "gadgets", "hardware"],
    "finance":           ["personalfinance", "investing", "wallstreetbets", "financialindependence"],
    "health":            ["health", "fitness", "nutrition", "loseit"],
    "gaming":            ["gaming", "pcgaming", "games", "IndieGaming"],
    "animation":         ["animation", "AnimationCareer", "learnanimation", "blender"],
    "history":           ["history", "AskHistorians", "HistoryMemes", "worldhistory"],
    "conspiracy":        ["conspiracy", "conspiracy_commons", "highstrangeness"],
    "spiritual":         ["spirituality", "Meditation", "awakened", "metaphysics"],
    "default":           ["videos", "youtube", "OutOfTheLoop", "TIL"],
})


def _scrape_reddit(queries: list, niche: str) -> list:
    """
    Reddit hot posts — free, no key.
    Uses the proper Reddit API User-Agent format:
      <platform>:<app ID>:<version> (by u/<reddit_username>)
    This prevents soft-blocking.
    """
    log.info("🟠  Reddit scrape...")
    headers = {
        "User-Agent": (
            "python:toontalkstudios.seo_sniper:v1.1 "
            "(by u/toontalkstudios; research only)"
        )
    }
    results = []
    niche_lower = niche.lower()
    subs = next(
        (v for k, v in _SUBREDDITS.items() if k in niche_lower),
        _SUBREDDITS.get("default", []),
    )

    endpoints = [
        f"https://www.reddit.com/search.json?q={requests.utils.quote(q)}&sort=hot&limit=8&t=week"
        for q in queries[:3]
    ] + [
        f"https://www.reddit.com/r/{sub}/hot.json?limit=8"
        for sub in subs[:3]
    ]

    delay = float(CFG.get("reddit_scrape", {}).get("delay_between_requests_s", 0.8))

    for url in endpoints:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                posts = r.json().get("data", {}).get("children", [])
                for p in posts:
                    d = p.get("data", {})
                    if d.get("title"):
                        results.append({
                            "title":        d.get("title", ""),
                            "score":        d.get("score", 0),
                            "subreddit":    d.get("subreddit", ""),
                            "num_comments": d.get("num_comments", 0),
                            "url":          f"https://reddit.com{d.get('permalink', '')}",
                        })
            elif r.status_code == 429:
                log.warning("Reddit rate-limited (429) — skipping remaining endpoints")
                break
            else:
                log.debug("Reddit %s returned %d", url[:60], r.status_code)
            time.sleep(delay)
        except Exception as e:
            log.debug("Reddit endpoint error: %s", e)

    results.sort(key=lambda x: x["score"], reverse=True)
    seen, unique = set(), []
    for item in results:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)
    return unique[:18]


def _scrape_youtube_api(queries: list) -> list:
    """YouTube Data API v3 — free 10k units/day."""
    yt_key = os.getenv("YOUTUBE_API_KEY", "")
    if not yt_key:
        log.info("⚠️  No YOUTUBE_API_KEY — YouTube trend data skipped")
        return []
    log.info("🔴  YouTube Data API...")
    results = []
    cutoff  = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    for q in queries[:3]:
        try:
            params = {
                "part": "snippet", "q": q, "type": "video",
                "order": "viewCount", "publishedAfter": cutoff,
                "maxResults": 8, "key": yt_key,
                "videoDuration": "medium", "relevanceLanguage": "en",
            }
            r = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params=params, timeout=15,
            )
            r.raise_for_status()
            for item in r.json().get("items", []):
                s   = item.get("snippet", {})
                vid = item.get("id", {}).get("videoId", "")
                results.append({
                    "title":       s.get("title", ""),
                    "channel":     s.get("channelTitle", ""),
                    "published":   s.get("publishedAt", "")[:10],
                    "description": s.get("description", "")[:180],
                    "url":         f"https://youtu.be/{vid}",
                })
            time.sleep(0.4)
        except Exception as e:
            log.warning("YouTube API error for '%s': %s", q, e)

    return results[:20]


def _scrape_google_trends(keywords: list) -> dict:
    """pytrends — Google Trends for YouTube searches."""
    if not HAS_PYTRENDS or not keywords:
        return {"available": False}
    log.info("📈  Google Trends...")
    out = {
        "available":         True,
        "interest_scores":   {},
        "rising_queries":    [],
        "top_queries":       [],
        "trending_searches": [],
    }
    try:
        # Brief pause before pytrends — reduces Google 429 (automation detection).
        # Google sees rapid-fire requests from the same IP as bot traffic.
        time.sleep(3)
        pt  = TrendReq(hl="en-US", tz=360, timeout=(10, 30))
        kws = [k for k in keywords[:5] if k]
        if not kws:
            return {"available": False}
        pt.build_payload(kws, cat=0, timeframe="now 7-d", geo="", gprop="youtube")

        iot = pt.interest_over_time()
        if not iot.empty:
            for kw in kws:
                if kw in iot.columns:
                    out["interest_scores"][kw] = int(iot[kw].mean())

        rq = pt.related_queries()
        for kw in kws:
            if kw in rq:
                rising = rq[kw].get("rising")
                top    = rq[kw].get("top")
                if rising is not None and not rising.empty:
                    out["rising_queries"].extend(rising["query"].tolist()[:6])
                if top is not None and not top.empty:
                    out["top_queries"].extend(top["query"].tolist()[:6])

        try:
            ts = pt.trending_searches(pn="united_states")
            out["trending_searches"] = ts[0].tolist()[:25]
        except Exception:
            pass

    except Exception as e:
        log.warning("Google Trends error: %s", e)
        out["available"] = False
        out["error"]     = str(e)
    return out


def _scrape_ddg_news(queries: list) -> list:
    """
    DuckDuckGo News — free, no key.
    Deduplicates queries (case-insensitive) to avoid instant 403 from
    hitting the same topic twice in one session.
    """
    if not HAS_DDG:
        return []
    log.info("🦆  DuckDuckGo News...")
    results = []

    # Deduplicate queries (the 403 in v1.0 was caused by "Toxic Relationships"
    # and "toxic relationships" being treated as separate queries)
    seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in queries[:3]:
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            unique_queries.append(q)

    try:
        ddgs = DDGS()
        for q in unique_queries[:2]:
            try:
                for item in ddgs.news(q, max_results=8):
                    results.append({
                        "title":  item.get("title",  ""),
                        "source": item.get("source", ""),
                        "date":   item.get("date",   ""),
                        "url":    item.get("url",    ""),
                    })
                time.sleep(2.5)   # 2.5s between queries — DDG 403s at <1s intervals
            except Exception as e:
                log.warning("DDG query '%s' failed: %s", q, e)
                break   # stop on first error to avoid cascading 403s
    except Exception as e:
        log.warning("DDG init error: %s", e)

    return results[:20]


def agent_collect_trends(script_data: dict, use_cache: bool = True) -> dict:
    """
    Orchestrate all 4 trend sources — runs them in parallel via
    ThreadPoolExecutor so total Agent-2 time ≈ slowest single source
    instead of the sum of all four.
    """
    log.info("🌐  [Agent 2/5] Trend Intelligence — scraping (parallel)...")

    main_topic = script_data.get("main_topic", "")
    niche      = script_data.get("niche", "")
    seed_kws   = script_data.get("seed_keywords", [])
    subtopics  = script_data.get("subtopics",    [])

    queries = list(dict.fromkeys(
        q for q in ([main_topic] + seed_kws[:3] + subtopics[:2]) if q
    ))[:5]

    # Check cache
    if use_cache and queries:
        key    = _cache_key(queries, niche)
        cached = _cache_read(key)
        if cached:
            return cached

    # ── All 4 scrapers run at the same time ───────────────────────────
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_reddit  = pool.submit(_scrape_reddit,        queries, niche)
        f_yt      = pool.submit(_scrape_youtube_api,   queries)
        f_trends  = pool.submit(_scrape_google_trends, seed_kws[:5] or [main_topic])
        f_ddg     = pool.submit(_scrape_ddg_news,      queries)

        reddit   = f_reddit.result()
        yt_vids  = f_yt.result()
        gtrends  = f_trends.result()
        ddg_news = f_ddg.result()

    result = {
        "queries_used":     queries,
        "reddit":           reddit,
        "youtube_trending": yt_vids,
        "google_trends":    gtrends,
        "news_headlines":   ddg_news,
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "_source_counts": {
            "reddit":  len(reddit),
            "youtube": len(yt_vids),
            "news":    len(ddg_news),
        },
    }

    if use_cache and queries:
        _cache_write(key, result)

    return result


# ══════════════════════════════════════════════════════════════════════
# AGENT 3 — SEO GENERATOR
# ══════════════════════════════════════════════════════════════════════

_SEO_SYS = (
    "You are a senior YouTube SEO strategist with 10+ years growing channels "
    "to millions of subscribers. You understand CTR psychology, YouTube's search "
    "algorithm, keyword clustering, and tag architecture. "
    "Respond with valid JSON only — no markdown, no preamble."
)


def agent_generate_seo(script_data: dict, trends: dict) -> dict:
    log.info("🎯  [Agent 3/5] SEO Generator — crafting titles, tags, description...")

    reddit_titles = [r["title"] for r in trends.get("reddit",           [])[:6]]
    yt_titles     = [v["title"] for v in trends.get("youtube_trending", [])[:6]]
    rising_qry    = trends.get("google_trends", {}).get("rising_queries", [])
    top_qry       = trends.get("google_trends", {}).get("top_queries",    [])
    news_titles   = [n.get("title", "") for n in trends.get("news_headlines", [])[:5]]

    prompt = f"""Generate complete YouTube SEO optimization for this video.

== SCRIPT ANALYSIS ==
Topic        : {script_data.get("main_topic")}
Niche        : {script_data.get("niche")} / {script_data.get("sub_niche")}
Content Type : {script_data.get("content_type")}
Core Message : {script_data.get("core_message")}
Seed Keywords: {script_data.get("seed_keywords", [])}
Subtopics    : {script_data.get("subtopics", [])}
Unique Angle : {script_data.get("unique_angle")}
Audience     : {script_data.get("target_audience")}

== LIVE TREND DATA ==
Reddit hot titles      : {reddit_titles}
YouTube trending titles: {yt_titles}
Rising Google queries  : {rising_qry}
Top Google queries     : {top_qry}
Recent news            : {news_titles}

== INSTRUCTIONS ==
- Primary title: under 70 chars, front-load main keyword, use power words
- Description: 450-500 words, keyword-dense first 150 chars, include timestamp skeleton
- Generate exactly 40 tags (mix: broad/specific/trending/long-tail)
- Include competitor-gap analysis for tags competitors miss

Return EXACTLY this JSON structure:
{{
  "titles": {{
    "primary": "best title under 70 chars",
    "alternatives": [
      "curiosity-gap formula variant",
      "number/list formula variant e.g. '7 Signs...'",
      "how-to formula variant",
      "shock/controversy formula variant",
      "trending-keyword-first variant"
    ],
    "power_words_used": ["list of power words in the primary title"],
    "title_score": 85,
    "title_analysis": "brief explanation of why this title works"
  }},
  "description": {{
    "full_text": "Full 450-500 word SEO description. First sentence must contain primary keyword. Use natural keyword integration. End with 3-5 relevant hashtags.",
    "first_line_hook": "First 150 chars only — must be compelling and keyword-rich",
    "timestamps": [
      "00:00 - Introduction",
      "01:30 - [main section 1]",
      "04:00 - [main section 2]",
      "07:00 - [main section 3]",
      "09:30 - Conclusion"
    ],
    "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],
    "cta_elements": [
      "subscribe prompt text",
      "comment engagement question",
      "like request text"
    ]
  }},
  "tags": {{
    "primary_tags":        ["5 exact-match high-priority tags"],
    "secondary_tags":      ["10 related broad topic tags"],
    "long_tail_tags":      ["10 specific 3-5 word phrases viewers actually search"],
    "trending_tags":       ["5 tags from current trending data above"],
    "competitor_gap_tags": ["5 tags competitors in this niche consistently miss"],
    "all_tags":            ["complete merged deduplicated list of 40 tags, priority ordered"]
  }},
  "keywords": {{
    "primary":         ["3-5 main target keywords"],
    "secondary":       ["8-10 supporting keywords"],
    "lsi":             ["8 latent semantic indexing keywords for natural density"],
    "long_tail":       ["5 long-tail 4-6 word search phrases"],
    "volume_estimate": {{"keyword_example": "high|medium|low"}}
  }},
  "competitor_analysis": {{
    "content_gap":                 "what competitors in this niche consistently fail to cover",
    "angle_advantage":             "why this video's angle outperforms existing content",
    "recommended_differentiators": ["differentiator 1", "differentiator 2", "differentiator 3"]
  }}
}}"""

    result = _llm.json_call(MODEL_SEO, _SEO_SYS, prompt, max_tokens=2500)
    return result if result else {}


# ══════════════════════════════════════════════════════════════════════
# AGENT 4 — VIRAL PREDICTOR
# ══════════════════════════════════════════════════════════════════════

_VIRAL_SYS = (
    "You are a viral content analyst who has studied 50,000+ YouTube videos "
    "across all niches. You understand psychological triggers, algorithm signals, "
    "and timing factors that determine virality. You are brutally honest about "
    "viral potential and give specific, actionable recommendations. "
    "Respond with valid JSON only — no markdown, no preamble."
)


def agent_predict_viral(script_data: dict, seo_data: dict, trends: dict) -> dict:
    log.info("🚀  [Agent 4/5] Viral Predictor — scoring viral potential...")

    primary_title  = seo_data.get("titles", {}).get("primary", "N/A")
    viral_tags     = seo_data.get("tags",   {}).get("trending_tags", [])
    rising_queries = trends.get("google_trends", {}).get("rising_queries", [])
    reddit_scores  = [(r["title"], r["score"]) for r in trends.get("reddit", [])[:5]]

    prompt = f"""Analyze this YouTube video's viral potential with brutal honesty.

== VIDEO METADATA ==
Topic            : {script_data.get("main_topic")}
Niche            : {script_data.get("niche")}
Primary Title    : {primary_title}
Core Message     : {script_data.get("core_message")}
Unique Angle     : {script_data.get("unique_angle")}
Emotional Hooks  : {script_data.get("emotional_triggers", [])}
Duration (est.)  : {script_data.get("estimated_duration_mins")} min
Content Type     : {script_data.get("content_type")}

== TREND ALIGNMENT ==
Rising searches  : {rising_queries}
Hot Reddit posts : {reddit_scores}
Trending tags    : {viral_tags}

== SCORING GUIDE (be honest, not flattering) ==
- trend_alignment:  0-20  (how well topic aligns with what's trending RIGHT NOW)
- title_strength:   0-20  (CTR potential of primary title — brutal assessment)
- emotional_impact: 0-20  (strength of emotional triggers in content)
- search_demand:    0-20  (search volume + competition ratio)
- timing_score:     0-20  (how timely/seasonal this topic is)
- Total = viral_score (0-100)

Return EXACTLY this JSON:
{{
  "viral_score": 78,
  "confidence": "high|medium|low",
  "verdict": "one-sentence brutal honest assessment",
  "score_breakdown": {{
    "trend_alignment":  14,
    "title_strength":   16,
    "emotional_impact": 18,
    "search_demand":    15,
    "timing_score":     15,
    "analysis": "2-3 sentence explanation of the scores"
  }},
  "viral_angles": [
    {{"angle": "specific angle name", "why_it_works": "psychology/trend reason", "implementation": "how to use this angle"}},
    {{"angle": "...", "why_it_works": "...", "implementation": "..."}},
    {{"angle": "...", "why_it_works": "...", "implementation": "..."}}
  ],
  "hook_suggestions": [
    {{"type": "question",   "script": "First 5-second hook using shocking question", "psychological_trigger": "curiosity/fear/etc"}},
    {{"type": "bold_claim", "script": "First 5-second hook with bold statement",    "psychological_trigger": "..."}},
    {{"type": "story",      "script": "First 5-second hook using story-in-motion",  "psychological_trigger": "..."}}
  ],
  "thumbnail_text_ideas": [
    {{"headline": "MAIN OVERLAY TEXT (3-5 words)", "subtext": "secondary text", "color_scheme": "dark red + white|black + yellow|etc", "style": "shock|curiosity|number|warning"}},
    {{"headline": "...", "subtext": "...", "color_scheme": "...", "style": "..."}},
    {{"headline": "...", "subtext": "...", "color_scheme": "...", "style": "..."}}
  ],
  "best_upload_times": {{
    "best_days":   ["Thursday", "Friday"],
    "time_window": "14:00–17:00 UTC",
    "avoid_days":  ["Monday", "Tuesday"],
    "reasoning":   "why these times work for this niche"
  }},
  "predicted_performance": {{
    "views_48h":       "500–2,000",
    "views_7d":        "2,000–8,000",
    "ctr_estimate":    "4–7%",
    "avg_watch_time":  "45–60%",
    "growth_catalyst": "what could push this to 10x views"
  }},
  "critical_improvements": [
    "most important change to increase viral potential",
    "second most important change",
    "third most important change"
  ],
  "risk_factors": [
    "potential reason this might underperform",
    "second risk factor"
  ]
}}"""

    result = _llm.json_call(MODEL_VIRAL, _VIRAL_SYS, prompt, max_tokens=2000)
    return result if result else {}


# ══════════════════════════════════════════════════════════════════════
# AGENT 5 — FUTURE IDEAS  (split into 2 parallel sub-calls)
# ══════════════════════════════════════════════════════════════════════
# v1.0 problem: one 3500-token call timed out twice → content_strategy: {}
# v1.1 fix:    split into sub-call A (~2500 tok) + sub-call B (~2000 tok)
#              run them in parallel → each finishes fast, no timeout risk.

_IDEAS_SYS = (
    "You are a YouTube channel growth strategist who has scaled dozens of channels "
    "from zero to 100k+ subscribers. You identify untapped content opportunities, "
    "emerging trends before they peak, and content series that maximize subscriber "
    "retention. You think 3–6 months ahead. "
    "Respond with valid JSON only — no markdown, no preamble."
)


def agent_generate_ideas(script_data: dict, trends: dict, seo_data: dict) -> dict:
    log.info("💡  [Agent 5/5] Future Ideas — 2 parallel sub-calls...")

    rising      = trends.get("google_trends", {}).get("rising_queries", [])
    hot_reddit  = [r["title"] for r in trends.get("reddit", [])[:5]]
    primary_kws = seo_data.get("keywords", {}).get("primary", [])

    ctx = f"""== CHANNEL CONTEXT ==
Niche           : {script_data.get("niche")} / {script_data.get("sub_niche")}
Content Pillars : {script_data.get("content_pillars", [])}
Current Topic   : {script_data.get("main_topic")}
Target Audience : {script_data.get("target_audience")}
Content Type    : {script_data.get("content_type")}

== TREND SIGNALS ==
Rising searches : {rising}
Hot Reddit posts: {hot_reddit}
Primary keywords: {primary_kws}"""

    # ── Sub-call A: 10 video ideas + 3 Shorts ─────────────────────────
    prompt_a = f"""{ctx}

Generate 10 diverse future video ideas and 3 Shorts ideas.

Return EXACTLY this JSON:
{{
  "future_video_ideas": [
    {{
      "title":                 "compelling optimized video title",
      "topic":                 "what this video covers",
      "viral_angle":           "specific reason this will perform well",
      "target_keyword":        "main keyword to optimize for",
      "content_format":        "educational|story|list|tutorial|case_study|debate",
      "estimated_viral_score": 80,
      "priority":              "high|medium|low",
      "search_volume":         "high|medium|low",
      "competition_level":     "high|medium|low",
      "suggested_timing":      "Month YYYY",
      "hook_idea":             "one-sentence hook for this video"
    }},
    ... (10 ideas total, all with different formats)
  ],
  "shorts_ideas": [
    {{"title": "Short title under 50 chars", "hook": "first 3-second script", "concept": "what it covers"}},
    {{"title": "...", "hook": "...", "concept": "..."}},
    {{"title": "...", "hook": "...", "concept": "..."}}
  ]
}}"""

    # ── Sub-call B: series + strategy + calendar ───────────────────────
    prompt_b = f"""{ctx}

Create a content strategy: series plan, niche gaps, cross-platform repurposing,
4-week content calendar, and 3 trend predictions.

Return EXACTLY this JSON:
{{
  "series_potential": {{
    "recommended":    true,
    "series_name":    "compelling series name",
    "concept":        "series concept in one sentence",
    "episode_titles": ["Episode 1 title", "Episode 2 title", "Episode 3 title", "Episode 4 title"],
    "why_series_wins":"why a series format grows the channel faster here"
  }},
  "niche_gaps": [
    {{"gap": "underserved topic", "opportunity": "why no one covers it and why you should"}},
    {{"gap": "...", "opportunity": "..."}},
    {{"gap": "...", "opportunity": "..."}}
  ],
  "cross_platform_repurposing": {{
    "tiktok":          {{"angle": "...", "hook": "...", "hashtags": ["#tag1", "#tag2"]}},
    "instagram":       {{"reel_hook": "...", "caption_idea": "..."}},
    "twitter_thread":  {{"first_tweet": "...", "thread_outline": ["tweet 1", "tweet 2", "tweet 3"]}},
    "newsletter":      "angle for email list — what makes this work as written content"
  }},
  "content_calendar": [
    {{"week": 1, "long_form_title": "...", "short_title": "...", "reasoning": "why this week"}},
    {{"week": 2, "long_form_title": "...", "short_title": "...", "reasoning": "..."}},
    {{"week": 3, "long_form_title": "...", "short_title": "...", "reasoning": "..."}},
    {{"week": 4, "long_form_title": "...", "short_title": "...", "reasoning": "..."}}
  ],
  "trend_predictions": [
    {{"trend": "emerging topic to watch", "expected_peak": "Month YYYY", "action": "what to create and when"}},
    {{"trend": "...", "expected_peak": "...", "action": "..."}},
    {{"trend": "...", "expected_peak": "...", "action": "..."}}
  ]
}}"""

    # Run both sub-calls in parallel (they're independent)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_a = pool.submit(_llm.json_call, MODEL_IDEAS, _IDEAS_SYS, prompt_a, 2000)
        f_b = pool.submit(_llm.json_call, MODEL_IDEAS, _IDEAS_SYS, prompt_b, 1500)
        result_a = f_a.result()
        result_b = f_b.result()

    merged: dict = {}
    if result_a: merged.update(result_a)
    if result_b: merged.update(result_b)

    if not merged:
        log.warning("Agent 5 returned empty — content strategy unavailable")
    return merged


# ══════════════════════════════════════════════════════════════════════
# OUTPUT ASSEMBLER
# ══════════════════════════════════════════════════════════════════════

def assemble_output(script_path: str, script_data: dict, trends: dict,
                    seo: dict, viral: dict, ideas: dict) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    quick = {
        "primary_title":      seo.get("titles", {}).get("primary", "N/A"),
        "viral_score":        viral.get("viral_score", 0),
        "verdict":            viral.get("verdict", ""),
        "top_5_tags":         seo.get("tags", {}).get("all_tags", [])[:5],
        "first_line_hook":    seo.get("description", {}).get("first_line_hook", ""),
        "best_upload_day":    (viral.get("best_upload_times", {}).get("best_days") or ["N/A"])[0],
        "upload_window":      viral.get("best_upload_times", {}).get("time_window", ""),
        "next_video_idea":    (ideas.get("future_video_ideas") or [{}])[0].get("title", "N/A"),
        "predicted_views_7d": viral.get("predicted_performance", {}).get("views_7d", "N/A"),
    }

    output = {
        "_meta": {
            "tool":        "YouTubeSEO Sniper v1.1",
            "author":      "ToonTalkStudios / Nour Fawzy",
            "github":      "https://github.com/NourMohammedF",
            "provider":    _llm.provider,
            "analyzed_at": now,
            "script_file": Path(script_path).name,
            "models_used": {
                "script_parser":  MODEL_PARSE,
                "seo_generator":  MODEL_SEO,
                "viral_predictor":MODEL_VIRAL,
                "ideas_agent":    MODEL_IDEAS,
            },
            "trend_sources": {
                "reddit":        trends.get("_source_counts", {}).get("reddit",  0),
                "youtube_api":   trends.get("_source_counts", {}).get("youtube", 0),
                "google_trends": trends.get("google_trends", {}).get("available", False),
                "news_articles": trends.get("_source_counts", {}).get("news",    0),
            },
        },
        "quick_summary":      quick,
        "script_analysis":    script_data,
        "trend_intelligence": {
            "queries_used":      trends.get("queries_used",      []),
            "reddit_hot":        trends.get("reddit",            [])[:10],
            "youtube_competing": trends.get("youtube_trending",  [])[:10],
            "google_trends":     trends.get("google_trends",     {}),
            "news_context":      trends.get("news_headlines",    [])[:10],
        },
        "seo_optimization":   seo,
        "viral_analysis":     viral,
        "content_strategy":   ideas,
    }

    return _validate_output(output)


# ══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

_BANNER = """
╔══════════════════════════════════════════════════════╗
║   YouTubeSEO Sniper v1.1  ·  ToonTalkStudios        ║
╚══════════════════════════════════════════════════════╝"""


def main():
    parser = argparse.ArgumentParser(
        description="YouTubeSEO Sniper v1.1 — AI-powered SEO & viral analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python seo_sniper.py script.txt
  python seo_sniper.py script.txt -o ./out/video_seo.json
  python seo_sniper.py script.txt --no-trends          (AI-only, fastest)
  python seo_sniper.py script.txt --no-cache           (force fresh scrape)
  python seo_sniper.py script.txt --provider groq      (use Groq Llama-70B free)
  python seo_sniper.py script.txt --provider openrouter (Qwen3-235B free)
  python seo_sniper.py script.txt --skip reddit news
        """,
    )
    parser.add_argument("script",        help="Path to video script (.txt)")
    parser.add_argument("-o", "--output", default="", help="Output JSON path")
    parser.add_argument("--no-trends",   action="store_true",
                        help="Skip all external trend scraping (AI-only mode)")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Force fresh trend scrape — ignore cache")
    parser.add_argument("--provider",    default="",
                        choices=["nim", "groq", "openrouter"],
                        help="LLM provider (default: reads config.json → 'nim')")
    parser.add_argument("--skip",        nargs="*", default=[],
                        choices=["reddit", "youtube", "trends", "news"],
                        help="Skip specific trend sources")
    args = parser.parse_args()

    # ── Provider init ─────────────────────────────────────────────────
    if args.provider:
        _reinit_client(args.provider)

    # ── Validate ──────────────────────────────────────────────────────
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"[ERROR] Script file not found: {script_path}")
        sys.exit(1)
    if not _llm.api_key:
        print(f"[ERROR] No API key for provider '{_llm.provider}'. "
              f"Set the correct env var in your .env file.")
        sys.exit(1)

    script_text = script_path.read_text(encoding="utf-8", errors="replace")
    out_path    = args.output or str(script_path.with_suffix("")) + "_seo.json"

    print(_BANNER)
    print(f"  Script   : {script_path.name}  ({len(script_text):,} chars)")
    print(f"  Output   : {out_path}")
    print(f"  Provider : {_llm.provider}")
    print(f"  Models   : {MODEL_PARSE.split('/')[-1]} / {MODEL_SEO.split('/')[-1]}")
    print()

    t0 = time.time()

    # ── Agent 1 ───────────────────────────────────────────────────────
    script_data = agent_parse_script(script_text)
    _partial_save(out_path, {"script_analysis": script_data})

    # ── Agent 2 ───────────────────────────────────────────────────────
    if args.no_trends:
        trends = {
            "queries_used": [], "reddit": [], "youtube_trending": [],
            "google_trends": {"available": False}, "news_headlines": [],
            "_source_counts": {"reddit": 0, "youtube": 0, "news": 0},
        }
    else:
        trends = agent_collect_trends(script_data, use_cache=not args.no_cache)
        if "reddit"  in args.skip: trends["reddit"]           = []
        if "youtube" in args.skip: trends["youtube_trending"] = []
        if "trends"  in args.skip: trends["google_trends"]    = {"available": False}
        if "news"    in args.skip: trends["news_headlines"]   = []
    _partial_save(out_path, {"script_analysis": script_data,
                              "trend_intelligence": trends})

    # ── Agent 3 ───────────────────────────────────────────────────────
    seo = agent_generate_seo(script_data, trends)
    _partial_save(out_path, {"script_analysis": script_data,
                              "trend_intelligence": trends,
                              "seo_optimization": seo})

    # ── Agent 4 ───────────────────────────────────────────────────────
    viral = agent_predict_viral(script_data, seo, trends)
    _partial_save(out_path, {"script_analysis": script_data,
                              "trend_intelligence": trends,
                              "seo_optimization": seo,
                              "viral_analysis":  viral})

    # ── Agent 5 ───────────────────────────────────────────────────────
    ideas = agent_generate_ideas(script_data, trends, seo)

    # ── Final output ──────────────────────────────────────────────────
    output = assemble_output(str(script_path), script_data, trends, seo, viral, ideas)
    Path(out_path).write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - t0
    qs = output.get("quick_summary", {})

    print("\n" + "═" * 56)
    print(f"  ✅  Done in {elapsed:.1f}s   →   {out_path}")
    print("─" * 56)
    print(f"  🏆  TITLE       {qs.get('primary_title', 'N/A')}")
    print(f"  🔥  VIRAL SCORE {qs.get('viral_score', 0)}/100  — {qs.get('verdict', '')}")
    print(f"  🏷️   TOP TAG     {(qs.get('top_5_tags') or ['N/A'])[0]}")
    print(f"  📅  UPLOAD      {qs.get('best_upload_day','N/A')}  {qs.get('upload_window','')}")
    print(f"  📈  VIEWS (7d)  {qs.get('predicted_views_7d','N/A')}")
    print(f"  💡  NEXT IDEA   {str(qs.get('next_video_idea','N/A'))[:55]}")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    main()