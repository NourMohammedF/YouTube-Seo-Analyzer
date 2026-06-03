
r"""
╔══════════════════════════════════════════════════════════════════════╗
║   YouTubeSEO Sniper v1.0                                             ║
║   AI-Powered Viral SEO Analyzer, Tag & Description Generator         ║
╚══════════════════════════════════════════════════════════════════════╝

Input  : video script (.txt)
Output : <script_name>_seo.json

AI Engine  : NVIDIA NIM  (mistralai/mistral-small-4-119b-2603 / llama-3.3-70b)
Trend Data : YouTube Data API  ·  Google Trends  ·  Reddit  ·  DuckDuckGo News

Agents
──────
  1. ScriptParser     — extracts niche, topics, entities, emotional triggers
  2. TrendIntelligence — scrapes Reddit, YouTube API, Google Trends, DDG News
  3. SEOGenerator     — titles, description, 40+ tags, keyword clusters
  4. ViralPredictor   — viral score, hook scripts, thumbnail text, upload timing
  5. FutureIdeas      — 10 content ideas, series potential, cross-platform plan
"""

import os, sys, json, re, time, datetime, argparse, logging
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── optional deps ─────────────────────────────────────────────────────
try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

# ── env & config ──────────────────────────────────────────────────────
load_dotenv()

NIM_API_KEY  = os.getenv("NVIDIA_NIM_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
YT_API_KEY   = os.getenv("YOUTUBE_API_KEY", "")

_cfg_path = Path(__file__).parent / "config.json"
CFG = json.loads(_cfg_path.read_text()) if _cfg_path.exists() else {}

MODEL_PARSE  = CFG.get("model_parse",  "meta/llama-3.3-70b-instruct")
MODEL_SEO    = CFG.get("model_seo",    "mistralai/mistral-small-4-119b-2603")
MODEL_VIRAL  = CFG.get("model_viral",  "mistralai/mistral-small-4-119b-2603")
MODEL_IDEAS  = CFG.get("model_ideas",  "mistralai/mistral-small-4-119b-2603")

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seo_sniper")


# ══════════════════════════════════════════════════════════════════════
# NVIDIA NIM CLIENT
# ══════════════════════════════════════════════════════════════════════

def _nim_call(model: str, messages: list, max_tokens: int = 2048,
              temperature: float = 0.3, retries: int = 3) -> str:
    """Raw NIM API call — returns text or empty string on failure."""
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{NIM_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=90,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            log.warning(f"NIM HTTP {r.status_code} (attempt {attempt}/{retries}): {e}")
        except Exception as e:
            log.warning(f"NIM error (attempt {attempt}/{retries}): {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return ""


def nim_json(model: str, system: str, user: str,
             max_tokens: int = 3000) -> dict:
    """Call NIM and safely parse the JSON response."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    raw = _nim_call(model, messages, max_tokens)
    if not raw:
        return {}
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    # try full parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # try to pull first {...} block
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    log.error("JSON parse failed — returning empty dict")
    log.debug("Raw NIM output (500 chars): %s", raw[:500])
    return {}


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
    result = nim_json(MODEL_PARSE, _PARSE_SYS, prompt, 1800)
    if not result:
        log.warning("Script parsing fallback — minimal metadata")
        result = {
            "main_topic": Path().cwd().name,
            "niche": "general",
            "content_type": "educational",
            "seed_keywords": [],
            "subtopics": [],
            "emotional_triggers": [],
        }
    return result


# ══════════════════════════════════════════════════════════════════════
# AGENT 2 — TREND INTELLIGENCE  (multi-source, mostly free)
# ══════════════════════════════════════════════════════════════════════

# niche → most relevant subreddits
_SUBREDDITS = CFG.get("subreddit_map", {
    "dark psychology":   ["psychology", "manipulation", "socialengineering", "coercivecontrol"],
    "psychology":        ["psychology", "socialpsychology", "AskPsychology"],
    "ai":                ["artificial", "MachineLearning", "LocalLLaMA", "ChatGPT"],
    "technology":        ["technology", "Futurology", "gadgets"],
    "finance":           ["personalfinance", "investing", "wallstreetbets"],
    "health":            ["health", "fitness", "nutrition"],
    "gaming":            ["gaming", "pcgaming", "games"],
    "true crime":        ["TrueCrime", "criminalminds", "unresolvedmysteries"],
    "self improvement":  ["selfimprovement", "productivity", "getmotivated"],
    "default":           ["videos", "youtube", "OutOfTheLoop", "explainlikeimfive"],
})


def _scrape_reddit(queries: list, niche: str) -> list:
    """Search Reddit hot posts — free, no API key."""
    log.info("🟠  Reddit trend scrape...")
    headers = {"User-Agent": "SEOSniper/1.0 (research bot)"}
    results = []

    # pick subreddits
    niche_lower = niche.lower()
    subs = next(
        (v for k, v in _SUBREDDITS.items() if k in niche_lower),
        _SUBREDDITS["default"]
    )

    # search + subreddit hot
    endpoints = [
        f"https://www.reddit.com/search.json?q={requests.utils.quote(q)}&sort=hot&limit=8&t=week"
        for q in queries[:3]
    ] + [
        f"https://www.reddit.com/r/{sub}/hot.json?limit=8"
        for sub in subs[:3]
    ]

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
            time.sleep(0.8)
        except Exception as e:
            log.debug(f"Reddit endpoint error: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    # deduplicate by title
    seen, unique = set(), []
    for r in results:
        if r["title"] not in seen:
            seen.add(r["title"])
            unique.append(r)
    return unique[:18]


def _scrape_youtube_api(queries: list) -> list:
    """YouTube Data API v3 — free 10k units/day."""
    if not YT_API_KEY:
        log.info("⚠️   No YOUTUBE_API_KEY — YouTube API trend data skipped")
        return []
    log.info("🔴  YouTube Data API — searching trending videos...")
    results = []
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    for q in queries[:3]:
        try:
            params = {
                "part":             "snippet",
                "q":                q,
                "type":             "video",
                "order":            "viewCount",
                "publishedAfter":   cutoff,
                "maxResults":       8,
                "key":              YT_API_KEY,
                "videoDuration":    "medium",
                "relevanceLanguage":"en",
            }
            r = requests.get("https://www.googleapis.com/youtube/v3/search",
                             params=params, timeout=15)
            r.raise_for_status()
            for item in r.json().get("items", []):
                s  = item.get("snippet", {})
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
            log.warning(f"YouTube API error for '{q}': {e}")

    return results[:20]


def _scrape_google_trends(keywords: list) -> dict:
    """pytrends — Google Trends data for YouTube searches."""
    if not HAS_PYTRENDS or not keywords:
        return {"available": False}
    log.info("📈  Google Trends — fetching interest & rising queries...")
    out = {
        "available":        True,
        "interest_scores":  {},
        "rising_queries":   [],
        "top_queries":      [],
        "trending_searches":[],
    }
    try:
        pt = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=2, backoff_factor=0.5)
        kws = keywords[:5]
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
        log.warning(f"Google Trends error: {e}")
        out["error"] = str(e)
    return out


def _scrape_ddg_news(queries: list) -> list:
    """DuckDuckGo news — free, no key needed."""
    if not HAS_DDG:
        return []
    log.info("🦆  DuckDuckGo — fetching recent news headlines...")
    results = []
    try:
        with DDGS() as ddgs:
            for q in queries[:2]:
                for item in ddgs.news(q, max_results=8):
                    results.append({
                        "title":  item.get("title", ""),
                        "source": item.get("source", ""),
                        "date":   item.get("date", ""),
                        "url":    item.get("url", ""),
                    })
                time.sleep(0.5)
    except Exception as e:
        log.warning(f"DDG news error: {e}")
    return results[:20]


def agent_collect_trends(script_data: dict) -> dict:
    """Orchestrate all trend data sources."""
    log.info("🌐  [Agent 2/5] Trend Intelligence — scraping all sources...")
    main_topic = script_data.get("main_topic", "")
    niche      = script_data.get("niche", "")
    seed_kws   = script_data.get("seed_keywords", [])
    subtopics  = script_data.get("subtopics", [])

    queries = list(dict.fromkeys(
        [main_topic] + seed_kws[:3] + subtopics[:2]
    ))[:5]
    queries = [q for q in queries if q]

    reddit   = _scrape_reddit(queries, niche)
    yt_vids  = _scrape_youtube_api(queries)
    gtrends  = _scrape_google_trends(seed_kws[:5] or [main_topic])
    ddg_news = _scrape_ddg_news(queries[:2])

    return {
        "queries_used":          queries,
        "reddit":                reddit,
        "youtube_trending":      yt_vids,
        "google_trends":         gtrends,
        "news_headlines":        ddg_news,
        "collected_at":          datetime.datetime.utcnow().isoformat() + "Z",
        "_source_counts": {
            "reddit":   len(reddit),
            "youtube":  len(yt_vids),
            "news":     len(ddg_news),
        },
    }


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

    reddit_titles = [r["title"] for r in trends.get("reddit", [])[:6]]
    yt_titles     = [v["title"] for v in trends.get("youtube_trending", [])[:6]]
    rising_qry    = trends.get("google_trends", {}).get("rising_queries", [])
    top_qry       = trends.get("google_trends", {}).get("top_queries", [])
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
Reddit hot titles     : {reddit_titles}
YouTube trending titles: {yt_titles}
Rising Google queries : {rising_qry}
Top Google queries    : {top_qry}
Recent news           : {news_titles}

== INSTRUCTIONS ==
- Primary title: under 70 chars, front-load main keyword, use power words
- Description: 450-500 words, keyword-dense first 150 chars, include timestamp skeleton
- Generate 40 tags (mix: broad/specific/trending/long-tail)
- Include a competitor-gap analysis for tags competitors miss

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
    "primary_tags":     ["5 exact-match high-priority tags"],
    "secondary_tags":   ["10 related broad topic tags"],
    "long_tail_tags":   ["10 specific 3-5 word phrases viewers actually search"],
    "trending_tags":    ["5 tags from current trending data above"],
    "competitor_gap_tags": ["5 tags competitors in this niche consistently miss"],
    "all_tags":         ["complete merged deduplicated list of 40 tags, priority ordered"]
  }},
  "keywords": {{
    "primary":    ["3-5 main target keywords"],
    "secondary":  ["8-10 supporting keywords"],
    "lsi":        ["8 latent semantic indexing keywords for natural density"],
    "long_tail":  ["5 long-tail 4-6 word search phrases"],
    "volume_estimate": {{"keyword_example": "high|medium|low"}}
  }},
  "competitor_analysis": {{
    "content_gap": "what competitors in this niche consistently fail to cover",
    "angle_advantage": "why this video's angle outperforms existing content",
    "recommended_differentiators": ["differentiator 1", "differentiator 2", "differentiator 3"]
  }}
}}"""

    result = nim_json(MODEL_SEO, _SEO_SYS, prompt, 3500)
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
    viral_tags     = seo_data.get("tags", {}).get("trending_tags", [])
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
    {{"type": "question", "script": "First 5-second hook script using shocking question", "psychological_trigger": "curiosity/fear/etc"}},
    {{"type": "bold_claim", "script": "First 5-second hook with bold statement", "psychological_trigger": "..."}},
    {{"type": "story", "script": "First 5-second hook using story-in-motion technique", "psychological_trigger": "..."}}
  ],
  "thumbnail_text_ideas": [
    {{"headline": "MAIN OVERLAY TEXT (3-5 words)", "subtext": "secondary text", "color_scheme": "dark red + white|black + yellow|etc", "style": "shock|curiosity|number|warning"}},
    {{"headline": "...", "subtext": "...", "color_scheme": "...", "style": "..."}},
    {{"headline": "...", "subtext": "...", "color_scheme": "...", "style": "..."}}
  ],
  "best_upload_times": {{
    "best_days":    ["Thursday", "Friday"],
    "time_window":  "14:00–17:00 UTC",
    "avoid_days":   ["Monday", "Tuesday"],
    "reasoning":    "why these times work for this niche"
  }},
  "predicted_performance": {{
    "views_48h":      "500–2,000",
    "views_7d":       "2,000–8,000",
    "ctr_estimate":   "4–7%",
    "avg_watch_time": "45–60%",
    "growth_catalyst":"what could push this to 10x views"
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

    result = nim_json(MODEL_VIRAL, _VIRAL_SYS, prompt, 2800)
    return result if result else {}


# ══════════════════════════════════════════════════════════════════════
# AGENT 5 — FUTURE IDEAS GENERATOR
# ══════════════════════════════════════════════════════════════════════

_IDEAS_SYS = (
    "You are a YouTube channel growth strategist who has scaled dozens of channels "
    "from zero to 100k+ subscribers. You identify untapped content opportunities, "
    "emerging trends before they peak, and content series that maximize subscriber "
    "retention. You think 3–6 months ahead. "
    "Respond with valid JSON only — no markdown, no preamble."
)

def agent_generate_ideas(script_data: dict, trends: dict, seo_data: dict) -> dict:
    log.info("💡  [Agent 5/5] Future Ideas — generating content strategy...")

    rising = trends.get("google_trends", {}).get("rising_queries", [])
    hot_reddit = [r["title"] for r in trends.get("reddit", [])[:5]]
    primary_kws = seo_data.get("keywords", {}).get("primary", [])

    prompt = f"""Create a 3-month future content strategy for this YouTube channel.

== CHANNEL CONTEXT ==
Niche           : {script_data.get("niche")} / {script_data.get("sub_niche")}
Content Pillars : {script_data.get("content_pillars", [])}
Current Topic   : {script_data.get("main_topic")}
Target Audience : {script_data.get("target_audience")}
Content Type    : {script_data.get("content_type")}

== TREND SIGNALS ==
Rising searches : {rising}
Hot Reddit posts: {hot_reddit}
Primary keywords: {primary_kws}

== INSTRUCTIONS ==
- Generate 10 future video ideas with high viral potential
- Each idea must be a DIFFERENT angle/format (no repeats)
- Include 3 Shorts ideas (60-sec vertical format)
- Identify niche gaps that no one is covering yet
- Create a 4-week content calendar

Return EXACTLY this JSON:
{{
  "future_video_ideas": [
    {{
      "title":                   "compelling optimized video title",
      "topic":                   "what this video covers",
      "viral_angle":             "specific reason this will perform well",
      "target_keyword":          "main keyword to optimize for",
      "content_format":          "educational|story|list|tutorial|case_study|debate",
      "estimated_viral_score":   80,
      "priority":                "high|medium|low",
      "search_volume":           "high|medium|low",
      "competition_level":       "high|medium|low",
      "suggested_timing":        "Month YYYY",
      "hook_idea":               "one-sentence hook for this video"
    }},
    ... (10 ideas total)
  ],
  "shorts_ideas": [
    {{"title": "Short title under 50 chars", "hook": "first 3 second script", "concept": "what the Short covers"}},
    {{"title": "...", "hook": "...", "concept": "..."}},
    {{"title": "...", "hook": "...", "concept": "..."}}
  ],
  "series_potential": {{
    "recommended": true,
    "series_name":     "compelling series name",
    "concept":         "series concept in one sentence",
    "episode_titles":  ["Episode 1 title", "Episode 2 title", "Episode 3 title", "Episode 4 title"],
    "why_series_wins": "why a series format grows the channel faster here"
  }},
  "niche_gaps": [
    {{"gap": "underserved topic", "opportunity": "why no one covers it and why you should"}},
    {{"gap": "...", "opportunity": "..."}},
    {{"gap": "...", "opportunity": "..."}}
  ],
  "cross_platform_repurposing": {{
    "tiktok":     {{"angle": "how to cut this for TikTok", "hook": "TikTok first line", "hashtags": ["#tag1", "#tag2"]}},
    "instagram":  {{"reel_hook": "first line for Reel", "caption_idea": "Instagram caption angle"}},
    "twitter_thread": {{"first_tweet": "thread opener tweet", "thread_outline": ["tweet 1 idea", "tweet 2 idea", "tweet 3 idea"]}},
    "newsletter": "angle for email list — what angle makes this work as written content"
  }},
  "content_calendar": [
    {{"week": 1, "long_form_title": "main video title", "short_title": "Short idea", "reasoning": "why this week"}},
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

    result = nim_json(MODEL_IDEAS, _IDEAS_SYS, prompt, 3500)
    return result if result else {}


# ══════════════════════════════════════════════════════════════════════
# OUTPUT ASSEMBLER
# ══════════════════════════════════════════════════════════════════════

def assemble_output(script_path: str, script_data: dict, trends: dict,
                    seo: dict, viral: dict, ideas: dict) -> dict:
    now = datetime.datetime.utcnow().isoformat() + "Z"

    quick = {
        "primary_title":    seo.get("titles", {}).get("primary", "N/A"),
        "viral_score":      viral.get("viral_score", 0),
        "verdict":          viral.get("verdict", ""),
        "top_5_tags":       seo.get("tags", {}).get("all_tags", [])[:5],
        "first_line_hook":  seo.get("description", {}).get("first_line_hook", ""),
        "best_upload_day":  (viral.get("best_upload_times", {}).get("best_days") or ["N/A"])[0],
        "upload_window":    viral.get("best_upload_times", {}).get("time_window", ""),
        "next_video_idea":  (ideas.get("future_video_ideas") or [{}])[0].get("title", "N/A"),
        "predicted_views_7d": viral.get("predicted_performance", {}).get("views_7d", "N/A"),
    }

    return {
        "_meta": {
            "tool":        "YouTubeSEO Sniper v1.0",
            "author":      "ToonTalkStudios / Nour Fawzy",
            "github":      "https://github.com/NourMohammedF",
            "analyzed_at": now,
            "script_file": Path(script_path).name,
            "models_used": {
                "script_parser": MODEL_PARSE,
                "seo_generator": MODEL_SEO,
                "viral_predictor":MODEL_VIRAL,
                "ideas_agent":   MODEL_IDEAS,
            },
            "trend_sources": {
                "reddit":         trends.get("_source_counts", {}).get("reddit", 0),
                "youtube_api":    trends.get("_source_counts", {}).get("youtube", 0),
                "google_trends":  trends.get("google_trends", {}).get("available", False),
                "news_articles":  trends.get("_source_counts", {}).get("news", 0),
            },
        },
        "quick_summary":       quick,
        "script_analysis":     script_data,
        "trend_intelligence": {
            "queries_used":            trends.get("queries_used", []),
            "reddit_hot":              trends.get("reddit", [])[:10],
            "youtube_competing":       trends.get("youtube_trending", [])[:10],
            "google_trends":           trends.get("google_trends", {}),
            "news_context":            trends.get("news_headlines", [])[:10],
        },
        "seo_optimization":    seo,
        "viral_analysis":      viral,
        "content_strategy":    ideas,
    }


# ══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

_BANNER = """
╔══════════════════════════════════════════════════════╗
║   YouTubeSEO Sniper                                  ║
╚══════════════════════════════════════════════════════╝"""

def main():
    parser = argparse.ArgumentParser(
        description="YouTubeSEO Sniper — AI-powered SEO & viral analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python seo_sniper.py script.txt
  python seo_sniper.py script.txt -o my_seo.json
  python seo_sniper.py script.txt --no-trends   (faster, AI only)
  python seo_sniper.py script.txt --skip reddit youtube
        """,
    )
    parser.add_argument("script",       help="Path to video script (.txt)")
    parser.add_argument("-o","--output",default="", help="Output JSON path")
    parser.add_argument("--no-trends",  action="store_true", help="Skip all external trend scraping")
    parser.add_argument("--skip",       nargs="*", default=[],
                        choices=["reddit","youtube","trends","news"],
                        help="Skip specific trend sources")
    args = parser.parse_args()

    # ── validate ──────────────────────────────────────────────────────
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"[ERROR] Script file not found: {script_path}")
        sys.exit(1)
    if not NIM_API_KEY:
        print("[ERROR] NVIDIA_NIM_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    # ── read script ───────────────────────────────────────────────────
    script_text = script_path.read_text(encoding="utf-8")
    out_path    = args.output or str(script_path.with_suffix("")) + "_seo.json"

    print(_BANNER)
    print(f"  Script  : {script_path.name}  ({len(script_text):,} chars)")
    print(f"  Output  : {out_path}")
    print(f"  Models  : {MODEL_PARSE.split('/')[-1]} / {MODEL_SEO.split('/')[-1]}")
    print()

    t0 = time.time()

    # ── run all 5 agents ──────────────────────────────────────────────
    script_data = agent_parse_script(script_text)

    if args.no_trends:
        trends = {
            "queries_used": [], "reddit": [], "youtube_trending": [],
            "google_trends": {"available": False}, "news_headlines": [],
            "_source_counts": {"reddit": 0, "youtube": 0, "news": 0},
        }
    else:
        trends = agent_collect_trends(script_data)
        if "reddit"  in args.skip: trends["reddit"]           = []
        if "youtube" in args.skip: trends["youtube_trending"] = []
        if "trends"  in args.skip: trends["google_trends"]    = {"available": False}
        if "news"    in args.skip: trends["news_headlines"]   = []

    seo   = agent_generate_seo(script_data, trends)
    viral = agent_predict_viral(script_data, seo, trends)
    ideas = agent_generate_ideas(script_data, trends, seo)

    output = assemble_output(str(script_path), script_data, trends, seo, viral, ideas)

    # ── save JSON ─────────────────────────────────────────────────────
    Path(out_path).write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - t0
    qs = output.get("quick_summary", {})

    print("\n" + "═"*56)
    print(f"  ✅  Done in {elapsed:.1f}s   →   {out_path}")
    print("─"*56)
    print(f"  🏆  TITLE       {qs.get('primary_title','N/A')}")
    print(f"  🔥  VIRAL SCORE {qs.get('viral_score', 0)}/100  — {qs.get('verdict','')}")
    print(f"  🏷️   TOP TAG     {(qs.get('top_5_tags') or ['N/A'])[0]}")
    print(f"  📅  UPLOAD      {qs.get('best_upload_day','N/A')}  {qs.get('upload_window','')}")
    print(f"  📈  VIEWS (7d)  {qs.get('predicted_views_7d','N/A')}")
    print(f"  💡  NEXT IDEA   {str(qs.get('next_video_idea','N/A'))[:55]}")
    print("═"*56 + "\n")


if __name__ == "__main__":
    main()
