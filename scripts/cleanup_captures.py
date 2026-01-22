import os
import time
import argparse

def cleanup(directory: str, older_than_hours: float, dry_run: bool) -> None:
    now = time.time()
    cutoff = now - (older_than_hours * 3600)

    if not os.path.isdir(directory):
        print(f"[INFO] Directory not found: {directory}")
        return

    deleted = 0
    kept = 0

    for name in os.listdir(directory):
        path = os.path.join(directory, name)

        # only clean wav files
        if not os.path.isfile(path) or not name.lower().endswith(".wav"):
            continue

        mtime = os.path.getmtime(path)
        if mtime < cutoff:
            if dry_run:
                print(f"[DRY] Would delete: {path}")
            else:
                os.remove(path)
                print(f"[DEL] {path}")
            deleted += 1
        else:
            kept += 1

    print(f"[DONE] deleted={deleted} kept={kept} dir={directory} older_than_hours={older_than_hours}")

def main():
    p = argparse.ArgumentParser(description="Cleanup old WAV files in captures/")
    p.add_argument("--dir", default="captures", help="Directory to clean (default: captures)")
    p.add_argument("--older-than-hours", type=float, default=24.0, help="Delete files older than N hours (default: 24)")
    p.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = p.parse_args()

    cleanup(args.dir, args.older_than_hours, args.dry_run)

if __name__ == "__main__":
    main()
