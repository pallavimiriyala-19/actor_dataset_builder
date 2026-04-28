import argparse
import json
import os
import sys

from src.pipeline import build_actor_dataset
from src.config import Config, configure_threading

configure_threading()


def _parse_line(line: str) -> tuple[str, str | None]:
    """Parse a list-file line. Format: 'Name' or 'Name | Wikipedia_Title'."""
    if "|" in line:
        parts = [p.strip() for p in line.split("|", 1)]
        return parts[0], (parts[1] or None)
    return line.strip(), None


def _read_list(path: str) -> list[tuple[str, str | None]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(_parse_line(line))
    return out


def main():
    p = argparse.ArgumentParser(description="Actor face dataset builder")
    p.add_argument("--actor", "-a", nargs="*", default=[], help="One or more actor names")
    p.add_argument("--list", "-l", help="Text file: one actor per line. Optional 'Name | Wikipedia_Title'.")
    p.add_argument("--max-images", type=int, default=None, help="Override max images per actor")
    args = p.parse_args()

    entries: list[tuple[str, str | None]] = [(n, None) for n in args.actor]
    if args.list:
        entries.extend(_read_list(args.list))

    seen = set()
    unique: list[tuple[str, str | None]] = []
    for name, wiki in entries:
        if name not in seen:
            seen.add(name)
            unique.append((name, wiki))

    if not unique:
        print("No actors provided. Use --actor or --list.", file=sys.stderr)
        sys.exit(2)

    cfg = Config()
    if args.max_images is not None:
        cfg.max_google_images = args.max_images

    summary = []
    for name, wiki_title in unique:
        try:
            meta = build_actor_dataset(name, cfg, wiki_title=wiki_title)
            summary.append(meta)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            summary.append({"name": name, "error": str(e)})

    os.makedirs(cfg.people_root, exist_ok=True)
    with open(os.path.join(cfg.people_root, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
