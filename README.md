# 🎯 YouTubeSEO Sniper v1.0
### AI-Powered Viral SEO Analyzer & Tag Generator

> Takes a video script → scrapes what's trending RIGHT NOW → outputs a complete JSON
> with viral-optimized titles, 40+ tags, full description, and 10 future content ideas.

---

## What It Does

```
script.txt
    │
    ▼  Agent 1 — Script Parser (llama-3.3-70b)
       Extracts: niche · topics · emotional triggers · seed keywords
    │
    ▼  Agent 2 — Trend Intelligence (free APIs)
       Scrapes: Reddit hot · YouTube trending · Google Trends · DDG News
    │
    ▼  Agent 3 — SEO Generator (mistral-small-4-119b)
       Generates: titles (×6) · 40 tags · 500-word description · keyword clusters
    │
    ▼  Agent 4 — Viral Predictor (mistral-small-4-119b)
       Scores: viral potential 0–100 · hook scripts · thumbnail text · upload timing
    │
    ▼  Agent 5 — Future Ideas (mistral-small-4-119b)
       Produces: 10 video ideas · 3 Shorts · series plan · 4-week calendar
    │
    ▼
script_seo.json 
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env:
#   NVIDIA_NIM_API_KEY=nvapi-xxx   (required — free at build.nvidia.com)
#   YOUTUBE_API_KEY=AIzaSy-xxx     (optional — free 10k units/day)
```

---

## Usage

### Single script
```bash
python seo_sniper.py my_script.txt
# Output: my_script_seo.json
```

### Custom output path
```bash
python seo_sniper.py my_script.txt -o ./output/video_01_seo.json
```

### Skip external scraping (AI-only, faster)
```bash
python seo_sniper.py my_script.txt --no-trends
```

### Skip specific sources
```bash
python seo_sniper.py my_script.txt --skip reddit trends
```

### Batch process a folder of scripts
```bash
python batch_analyze.py ./scripts/ --output-dir ./seo_output/
```

### Run the test script
```bash
python seo_sniper.py sample_script.txt
```

---

## ToonTalkStudios Integration

Drop `seo_sniper.py`, `pipeline_hook.py`, and `config.json` into your pipeline folder,
then call from your existing code:

```python
from pipeline_hook import run_seo_pipeline, inject_into_metadata

# After your script generation step:
seo = run_seo_pipeline(
    script_text=generated_script,
    video_title_hint="dark psychology",   # optional niche hint
    save_path="./output/video_seo.json",  # optional auto-save
)

# Inject into your existing metadata dict:
metadata = inject_into_metadata(existing_metadata, seo)
# metadata now has: title, tags, description, viral_score, thumbnail_texts, etc.

# Quick helpers:
from pipeline_hook import get_best_title, get_all_tags, get_viral_score

title = get_best_title(seo)          # → "The Dark Psychology of..."
tags  = get_all_tags(seo)            # → ["manipulation", "psychology", ...]
score = get_viral_score(seo)         # → 82
```

---

## Output JSON Structure

```json
{
  "_meta": { "tool", "analyzed_at", "models_used", "trend_sources" },

  "quick_summary": {
    "primary_title": "...",
    "viral_score": 82,
    "top_5_tags": ["..."],
    "best_upload_day": "Thursday",
    "predicted_views_7d": "3,000–9,000"
  },

  "script_analysis": {
    "main_topic", "niche", "sub_niche", "content_type", "tone",
    "emotional_triggers", "unique_angle", "seed_keywords",
    "target_audience", "estimated_duration_mins", ...
  },

  "trend_intelligence": {
    "reddit_hot": [ { "title", "score", "subreddit", "num_comments" } ],
    "youtube_competing": [ { "title", "channel", "url" } ],
    "google_trends": { "interest_scores", "rising_queries", "top_queries" },
    "news_context": [ { "title", "source", "date" } ]
  },

  "seo_optimization": {
    "titles": {
      "primary": "...",
      "alternatives": ["×5 formula variants"],
      "power_words_used": ["..."],
      "title_score": 88
    },
    "description": {
      "full_text": "450-500 word SEO description...",
      "first_line_hook": "first 150 chars",
      "timestamps": ["00:00 - Intro", ...],
      "hashtags": ["#tag1", ...]
    },
    "tags": {
      "primary_tags", "secondary_tags", "long_tail_tags",
      "trending_tags", "competitor_gap_tags",
      "all_tags": ["40 tags, priority ordered"]
    },
    "keywords": { "primary", "secondary", "lsi", "long_tail" },
    "competitor_analysis": { "content_gap", "angle_advantage" }
  },

  "viral_analysis": {
    "viral_score": 82,
    "verdict": "...",
    "score_breakdown": { "trend_alignment", "title_strength", ... },
    "viral_angles": [ { "angle", "why_it_works", "implementation" } ],
    "hook_suggestions": [ { "type", "script", "psychological_trigger" } ],
    "thumbnail_text_ideas": [ { "headline", "subtext", "color_scheme", "style" } ],
    "best_upload_times": { "best_days", "time_window", "reasoning" },
    "predicted_performance": { "views_48h", "views_7d", "ctr_estimate" },
    "critical_improvements": ["..."],
    "risk_factors": ["..."]
  },

  "content_strategy": {
    "future_video_ideas": [ { "title", "viral_angle", "priority", ... } × 10 ],
    "shorts_ideas": [ { "title", "hook", "concept" } × 3 ],
    "series_potential": { "series_name", "episode_titles", ... },
    "niche_gaps": [ { "gap", "opportunity" } ],
    "cross_platform_repurposing": { "tiktok", "instagram", "twitter_thread" },
    "content_calendar": [ { "week", "long_form_title", "short_title" } × 4 ],
    "trend_predictions": [ { "trend", "expected_peak", "action" } ]
  }
}
```

---

## Models Used

| Agent          | Model                              | Why                          |
|----------------|------------------------------------|------------------------------|
| Script Parser  | `meta/llama-3.3-70b-instruct`      | Deep semantic understanding  |
| SEO Generator  | `mistralai/mistral-small-4-119b`   | Fast + strong instruction following |
| Viral Predictor| `mistralai/mistral-small-4-119b`   | Top performer in Hermes bench |
| Ideas Agent    | `mistralai/mistral-small-4-119b`   | Best real-time chain speed   |

All configurable in `config.json`.

---

## API Keys Required

| Key                    | Required | Free Tier            | Where to get                        |
|------------------------|----------|----------------------|-------------------------------------|
| `NVIDIA_NIM_API_KEY`   | ✅ Yes   | Yes (rate limited)   | [build.nvidia.com](https://build.nvidia.com) |
| `YOUTUBE_API_KEY`      | ❌ No    | 10,000 units/day     | [Google Cloud Console](https://console.cloud.google.com) |

Reddit, Google Trends, and DuckDuckGo News are all **completely free** — no keys needed.

---

## Files

```
youtube_seo_sniper/
├── seo_sniper.py        ← Main pipeline (run this)
├── batch_analyze.py     ← Process folder of scripts
├── pipeline_hook.py     ← ToonTalkStudios integration module
├── config.json          ← Model selection + subreddit maps
├── requirements.txt
├── .env.example
└── sample_script.txt    ← Dark psychology demo script
```
