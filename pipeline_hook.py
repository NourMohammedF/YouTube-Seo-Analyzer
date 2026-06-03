"""
pipeline_hook.py  —  YouTubeSEO Sniper  ·  ToonTalkStudios Integration
───────────────────────────────────────────────────────────────────────
Drop-in module for the ToonTalkStudios autonomous video pipeline.
Call run_seo_pipeline() from your existing pipeline after script generation.

Integration example (in your existing pipeline):
  from pipeline_hook import run_seo_pipeline, inject_into_metadata

  # After script is generated:
  seo_data = run_seo_pipeline(script_text, video_title_hint="dark psychology")
  
  # Inject into your video metadata dict:
  metadata = inject_into_metadata(existing_metadata, seo_data)
"""

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("seo_hook")

# ── Import core agents ────────────────────────────────────────────────
try:
    from seo_sniper import (
        agent_parse_script,
        agent_collect_trends,
        agent_generate_seo,
        agent_predict_viral,
        agent_generate_ideas,
        assemble_output,
        NIM_API_KEY,
    )
    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False
    log.warning("seo_sniper.py not in path — pipeline_hook in stub mode")


def run_seo_pipeline(
    script_text:       str,
    video_title_hint:  str  = "",
    skip_trends:       bool = False,
    save_path:         Optional[str] = None,
) -> dict:
    """
    Run the full 5-agent SEO pipeline on a script string.
    
    Args:
        script_text     : Raw video script as string
        video_title_hint: Optional hint to prepend to script for better parsing
        skip_trends     : If True, skip external scraping (faster, offline mode)
        save_path       : If set, save JSON to this path

    Returns:
        Full SEO analysis dict (same structure as _seo.json output)
    """
    if not _AGENTS_AVAILABLE:
        log.error("seo_sniper.py not importable — returning empty dict")
        return {}

    if not NIM_API_KEY:
        log.error("NVIDIA_NIM_API_KEY not set")
        return {}

    # Prepend hint if provided
    full_script = f"[VIDEO TOPIC HINT: {video_title_hint}]\n\n{script_text}" if video_title_hint else script_text

    log.info("SEO pipeline started via pipeline_hook...")

    script_data = agent_parse_script(full_script)

    if skip_trends:
        trends = {
            "queries_used": [], "reddit": [], "youtube_trending": [],
            "google_trends": {"available": False}, "news_headlines": [],
            "_source_counts": {"reddit": 0, "youtube": 0, "news": 0},
        }
    else:
        trends = agent_collect_trends(script_data)

    seo   = agent_generate_seo(script_data, trends)
    viral = agent_predict_viral(script_data, seo, trends)
    ideas = agent_generate_ideas(script_data, trends, seo)

    output = assemble_output(
        video_title_hint or "pipeline_input",
        script_data, trends, seo, viral, ideas,
    )

    if save_path:
        Path(save_path).write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"SEO analysis saved to {save_path}")

    return output


def inject_into_metadata(existing_metadata: dict, seo_output: dict) -> dict:
    """
    Merge SEO output into your existing video metadata dict.
    Safe — only adds/updates SEO-related fields, doesn't overwrite unrelated keys.

    Args:
        existing_metadata : Your current video metadata dict
        seo_output        : Output from run_seo_pipeline()

    Returns:
        Updated metadata dict with SEO fields injected
    """
    seo_opt  = seo_output.get("seo_optimization", {})
    viral    = seo_output.get("viral_analysis", {})
    strategy = seo_output.get("content_strategy", {})
    quick    = seo_output.get("quick_summary", {})

    injected = dict(existing_metadata)  # shallow copy

    # Core YouTube upload fields
    injected["title"]       = seo_opt.get("titles", {}).get("primary", existing_metadata.get("title", ""))
    injected["title_alts"]  = seo_opt.get("titles", {}).get("alternatives", [])
    injected["description"] = seo_opt.get("description", {}).get("full_text", existing_metadata.get("description", ""))
    injected["tags"]        = seo_opt.get("tags", {}).get("all_tags", existing_metadata.get("tags", []))
    injected["hashtags"]    = seo_opt.get("description", {}).get("hashtags", [])

    # SEO metadata
    injected["keywords"]    = seo_opt.get("keywords", {}).get("primary", [])
    injected["lsi_keywords"]= seo_opt.get("keywords", {}).get("lsi", [])

    # Viral intelligence
    injected["viral_score"]     = viral.get("viral_score", 0)
    injected["viral_verdict"]   = viral.get("verdict", "")
    injected["hook_suggestions"]= viral.get("hook_suggestions", [])
    injected["thumbnail_texts"] = viral.get("thumbnail_text_ideas", [])
    injected["best_upload_time"]= viral.get("best_upload_times", {})
    injected["predicted_views"] = viral.get("predicted_performance", {})

    # Content strategy
    injected["future_ideas"]    = strategy.get("future_video_ideas", [])[:5]
    injected["shorts_ideas"]    = strategy.get("shorts_ideas", [])
    injected["content_calendar"]= strategy.get("content_calendar", [])

    # Timestamps for description
    injected["timestamps"]      = seo_opt.get("description", {}).get("timestamps", [])

    return injected


def get_best_title(seo_output: dict) -> str:
    """Quick helper — extract just the primary title."""
    return seo_output.get("seo_optimization", {}).get("titles", {}).get("primary", "")


def get_all_tags(seo_output: dict) -> list:
    """Quick helper — extract the full tag list."""
    return seo_output.get("seo_optimization", {}).get("tags", {}).get("all_tags", [])


def get_description(seo_output: dict) -> str:
    """Quick helper — extract the full description."""
    return seo_output.get("seo_optimization", {}).get("description", {}).get("full_text", "")


def get_viral_score(seo_output: dict) -> int:
    """Quick helper — extract viral score."""
    return seo_output.get("viral_analysis", {}).get("viral_score", 0)


# ── Standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    test_script = """
    The Dark Art of Mirroring: How Manipulators Use Your Reflection Against You.
    Have you ever met someone who seemed to understand you perfectly?
    They mirrored your speech, your gestures, your interests — until you trusted them completely.
    Today we expose how this works and how to detect it.
    """

    print("Running pipeline_hook test...")
    result = run_seo_pipeline(test_script, video_title_hint="dark psychology mirroring", skip_trends=True)

    if result:
        print(f"\nTitle: {get_best_title(result)}")
        print(f"Viral Score: {get_viral_score(result)}/100")
        print(f"Tags (first 5): {get_all_tags(result)[:5]}")
    else:
        print("Pipeline returned empty — check NIM API key")
