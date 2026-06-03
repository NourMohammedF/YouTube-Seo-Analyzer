
r"""
batch_analyze.py  —  YouTubeSEO Sniper  ·  ToonTalkStudios
────────────────────────────────────────────────────────────
Run seo_sniper.py on every .txt file in a folder.

Usage:
  python batch_analyze.py ./scripts/
  python batch_analyze.py ./scripts/ --output-dir ./seo_output/
  python batch_analyze.py ./scripts/ --no-trends --delay 5
"""

import os, sys, time, json, argparse, subprocess, logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")

SNIPER = Path(__file__).parent / "seo_sniper.py"


def run_single(script: Path, out_dir: Path, extra_args: list) -> dict:
    out_file = out_dir / (script.stem + "_seo.json")
    cmd = [sys.executable, str(SNIPER), str(script), "-o", str(out_file)] + extra_args

    log.info(f"Processing: {script.name}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    status = "✅ OK" if result.returncode == 0 else "❌ FAILED"
    return {
        "file":     script.name,
        "output":   str(out_file),
        "status":   status,
        "elapsed_s":round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch SEO analysis for multiple scripts")
    parser.add_argument("folder",       help="Folder containing .txt script files")
    parser.add_argument("--output-dir", default="", help="Folder for JSON outputs (default: same as input)")
    parser.add_argument("--no-trends",  action="store_true")
    parser.add_argument("--delay",      type=float, default=3.0,
                        help="Seconds to wait between scripts (avoid rate limits)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)

    scripts = sorted(folder.glob("*.txt"))
    if not scripts:
        print(f"[WARN] No .txt files found in {folder}")
        sys.exit(0)

    out_dir = Path(args.output_dir) if args.output_dir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = ["--no-trends"] if args.no_trends else []

    print(f"\n{'═'*50}")
    print(f"  Batch SEO Sniper  —  {len(scripts)} scripts found")
    print(f"  Output dir: {out_dir}")
    print(f"{'═'*50}\n")

    results = []
    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] ──────────────────────────────")
        r = run_single(script, out_dir, extra)
        results.append(r)
        if i < len(scripts):
            log.info(f"Waiting {args.delay}s before next script...")
            time.sleep(args.delay)

    # summary report
    report = {
        "batch_run_at": datetime.utcnow().isoformat() + "Z",
        "total":    len(results),
        "success":  sum(1 for r in results if "✅" in r["status"]),
        "failed":   sum(1 for r in results if "❌" in r["status"]),
        "results":  results,
    }
    report_path = out_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'═'*50}")
    print(f"  Batch Complete")
    print(f"  ✅ {report['success']} succeeded   ❌ {report['failed']} failed")
    print(f"  Report: {report_path}")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()
