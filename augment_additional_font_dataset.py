"""
Generate additional cropped text images per font family.

Defaults:
- Fonts: /home/ubuntu/fonts_top_1000 (families derived from filename prefix before '--')
- Backgrounds: /home/ubuntu/jjseol/layer_data/inpainting_250k_subset_rendered
- Layout JSON: /home/ubuntu/jjseol/layer_data/inpainting_250k_subset
- Output: /home/ubuntu/jjseol/additional_data/font_crops_additional_5000/<family>/<uuid>.jpg
- Target: 5000 new samples per family (resume-friendly; skips files that already exist)

The generation follows the notebook logic:
- Build text pools from layout JSON (cached to output dir)
- Optionally mix in Wikipedia sentences
- Compose text-free backgrounds by dropping text layers
- Render random text (font/color/position/stroke) then crop bbox and save
- Multithreaded per family
"""

import argparse
import concurrent.futures
import json
import os
import random
import textwrap
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import wikipedia
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ----------------------------
# Paths (override via CLI if needed)
# ----------------------------
FONTS_DIR = Path("/home/ubuntu/fonts_top_1000").resolve()
OUTPUT_DIR = Path("/home/ubuntu/jjseol/additional_data/font_crops_additional_5000").resolve()
RENDERED_ROOT = Path("/home/ubuntu/jjseol/layer_data/inpainting_250k_subset_rendered").resolve()
JSON_ROOT = Path("/home/ubuntu/jjseol/layer_data/inpainting_250k_subset").resolve()

# ----------------------------
# Text helpers (borrowed from notebook)
# ----------------------------


def get_font_files(directory: Path) -> List[Path]:
    preferred_exts = ("*.ttf", "*.otf")
    alt_exts = ("*.woff", "*.woff2")
    fonts: List[Path] = []
    for ext in preferred_exts:
        fonts.extend(directory.glob(ext))
    for ext in alt_exts:
        fonts.extend(directory.glob(ext))
    # dedupe while preserving order
    seen = set()
    unique_fonts = []
    for f in fonts:
        if f in seen:
            continue
        seen.add(f)
        unique_fonts.append(f)
    return unique_fonts


def build_font_families(directory: Path) -> Dict[str, List[Path]]:
    families: Dict[str, List[Path]] = {}
    for path in get_font_files(directory):
        stem = path.stem
        family = stem.split("--")[0] if "--" in stem else stem
        families.setdefault(family, []).append(path)
    return families


def _extract_text_strings(component: dict, max_chars: int = 200) -> List[str]:
    texts = []
    if not isinstance(component, dict):
        return texts
    comp_type = component.get("type")
    for key in ("text", "content", "value", "name"):
        val = component.get(key)
        if isinstance(val, str) and val.strip():
            val = val.strip()
            if len(val) <= max_chars:
                texts.append(val)
            break
    if comp_type == "GROUP":
        for child in component.get("components") or []:
            texts.extend(_extract_text_strings(child, max_chars=max_chars))
    return texts


def collect_text_lengths(json_root: Path, max_files: int = -1, max_chars: int = 200) -> Tuple[List[int], int, List[str]]:
    lengths: List[int] = []
    texts: List[str] = []
    total_texts = 0
    json_files = sorted(json_root.glob("*.json"))
    if max_files is not None and max_files > 0:
        json_files = json_files[:max_files]
    for jp in json_files:
        try:
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        comps = data.get("layout_config", {}).get("components", [])
        for comp in comps:
            for txt in _extract_text_strings(comp, max_chars=max_chars):
                total_texts += 1
                lengths.append(len(txt))
                texts.append(txt)
    return lengths, total_texts, texts


def load_or_build_length_pool(json_root: Path, cache_path: Path, max_files: int = -1, max_chars: int = 200) -> Tuple[List[int], int]:
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                vals = [int(line.strip()) for line in f if line.strip().isdigit()]
            if vals:
                print(f"Loaded text length cache from {cache_path} (n={len(vals)})")
                return vals, len(vals)
        except Exception:
            pass
    lengths, total, _ = collect_text_lengths(json_root, max_files=max_files, max_chars=max_chars)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            for v in lengths:
                f.write(f"{v}\n")
    except Exception:
        pass
    print(f"Built text length cache (n={len(lengths)}) from JSON (total texts {total})")
    return lengths, total


def load_or_build_text_pool(json_root: Path, cache_path: Path, max_files: int = -1, max_chars: int = 200) -> Tuple[List[str], int]:
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f if line.strip()]
            if lines:
                print(f"Loaded text cache from {cache_path} (n={len(lines)})")
                return lines, len(lines)
        except Exception:
            pass
    _, total, texts = collect_text_lengths(json_root, max_files=max_files, max_chars=max_chars)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            for t in texts:
                f.write(f"{t}\n")
    except Exception:
        pass
    print(f"Built text content cache (n={len(texts)}) from JSON (total texts {total})")
    return texts, total


def trim_text_to_length(text: str, target_len: int) -> str:
    if target_len <= 0:
        return text
    if len(text) <= target_len:
        return text
    return text[:target_len].rstrip()


def wiki_sentences(count=64, min_words=1, lang="en", max_chars=200, max_tries=20) -> List[str]:
    wikipedia.set_lang(lang)
    results: List[str] = []
    tries = 0
    while len(results) < count and tries < max_tries:
        tries += 1
        try:
            summary = wikipedia.page(wikipedia.random(1)).summary
        except (wikipedia.DisambiguationError, wikipedia.PageError, Exception):
            continue
        sentences = [s.strip() for s in summary.replace("\n", " ").split(". ") if s.strip()]
        for sent in sentences:
            words = sent.split()
            if len(words) < min_words:
                continue
            joined = " ".join(words)
            if len(joined) > max_chars:
                start = random.randint(0, len(joined) - max_chars)
                joined = joined[start : start + max_chars]
            if joined:
                results.append(joined)
            if len(results) >= count:
                break
    return results[:count]


def get_random_text(
    length_pool: List[int],
    json_text_pool: List[str],
    wiki_pool: List[str],
    wiki_ratio: float = 0.5,
    max_chars: int = 200,
) -> str:
    lengths = length_pool or [32, 48, 64, 80]
    target_len = min(max_chars, random.choice(lengths))

    use_wiki = wiki_pool and (random.random() < wiki_ratio)
    if use_wiki:
        t = random.choice(wiki_pool)
        return trim_text_to_length(t, target_len)
    if json_text_pool:
        t = random.choice(json_text_pool)
        return trim_text_to_length(t, target_len)
    fallback = [
        "Synthetic text overlay for dataset augmentation.",
        "Quick brown fox jumps over the lazy dog.",
        "Overlaying text on backgrounds for visual QA.",
        "Simple bbox demo without file writes.",
    ]
    return trim_text_to_length(random.choice(fallback), target_len)


# ----------------------------
# Background helpers
# ----------------------------


def load_image(path: Path, mode: str = "RGBA") -> Image.Image:
    with Image.open(path) as im:
        return im.convert(mode)


def resolve_background_path(sample_dir: Path) -> Path:
    direct = sample_dir / "background.png"
    if direct.exists():
        return direct
    prefixed = sample_dir / f"{sample_dir.name}_background.png"
    if prefixed.exists():
        return prefixed
    for cand in sorted(sample_dir.glob("*_background.png")):
        if "thumbnail" in cand.name.lower():
            continue
        return cand
    raise FileNotFoundError(f"Background image not found in {sample_dir}")


def find_component_paths(sample_dir: Path) -> List[Path]:
    patterns = ["component_*.png", f"{sample_dir.name}_component_*.png", "*_component_*.png"]
    indexed: List[Tuple[int, Path]] = []
    for pattern in patterns:
        for path in sample_dir.glob(pattern):
            if "thumbnail" in path.name.lower():
                continue
            for part in reversed(path.stem.split("_")):
                if part.isdigit():
                    indexed.append((int(part), path))
                    break
        if indexed:
            break
    indexed.sort(key=lambda x: x[0])
    return [p for _, p in indexed]


def load_rendered_sample(sample_dir: Path) -> Tuple[Image.Image, List[Image.Image]]:
    background = load_image(resolve_background_path(sample_dir), mode="RGBA")
    components = [load_image(p, mode="RGBA") for p in find_component_paths(sample_dir)]
    return background, components


def is_text_like_component(comp_cfg: dict) -> bool:
    comp_type = comp_cfg.get("type")
    if comp_type == "TEXT":
        return True
    if comp_type == "GROUP":
        children = comp_cfg.get("components") or []
        if children and all(child.get("type") == "TEXT" for child in children):
            return True
    return False


def compose_text_free_background(sample_dir: Path, json_root: Path) -> Tuple[Image.Image, Dict]:
    json_path = json_root / f"{sample_dir.name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Layout JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    layout_components = data.get("layout_config", {}).get("components", [])
    background, components = load_rendered_sample(sample_dir)
    composite = background.convert("RGBA")
    for idx, comp in enumerate(components):
        if idx >= len(layout_components):
            continue
        if is_text_like_component(layout_components[idx]):
            continue
        composite = Image.alpha_composite(composite, comp.convert("RGBA"))
    return composite.convert("RGB"), {"sample_dir": str(sample_dir), "json": str(json_path)}


def random_text_free_background(render_root: Path, json_root: Path, tries: int = 10, candidates: List[Path] = None, cache_paths: List[Path] = None) -> Tuple[Image.Image, Dict]:
    if cache_paths:
        img_path = random.choice(cache_paths)
        return load_image(img_path, mode="RGB"), {"sample_dir": str(img_path.parent), "json": "<cached>"}
    if candidates is None:
        candidates = [d for d in render_root.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No rendered sample dirs under {render_root}")
    last_err = None
    for _ in range(min(tries, len(candidates))):
        sample_dir = random.choice(candidates)
        try:
            return compose_text_free_background(sample_dir, json_root)
        except Exception as exc:
            last_err = exc
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Failed to build text-free background")


# ----------------------------
# Text rendering
# ----------------------------


def _cache_bg_one(sample_dir: Path, json_root: Path, out_dir: Path, idx: int) -> Path | None:
    """Compose text-free background and save to cache. Returns saved path or None on failure."""
    try:
        img, _ = compose_text_free_background(sample_dir, json_root)
        out_path = out_dir / f"{idx:06d}_{uuid.uuid4().hex}.jpg"
        img.save(out_path, format="JPEG", quality=90, subsampling=1, optimize=False)
        return out_path
    except Exception:
        return None


def try_load_font(font_paths: List[Path], size: int) -> Tuple[ImageFont.FreeTypeFont, Path]:
    shuffled = list(font_paths)
    random.shuffle(shuffled)
    for font_path in shuffled:
        try:
            return ImageFont.truetype(str(font_path), size), font_path
        except (OSError, UnidentifiedImageError):
            continue
        except Exception:
            continue
    return None, None


def make_augmented_image(
    background: Image.Image,
    text: str,
    font_paths: List[Path],
    font_size: int = 48,
    margin: int = 24,
    stroke_width: int = 2,
) -> Tuple[Image.Image, Dict]:
    bg = background.convert("RGB").copy()
    draw = ImageDraw.Draw(bg)

    font, font_path = try_load_font(font_paths, font_size)
    if font is None:
        raise RuntimeError("No usable font file found for this family")

    max_width = bg.width - 2 * margin
    wrapped = textwrap.fill(text, width=max(10, int(max_width / (font_size * 0.6))))

    tmp_bbox = draw.textbbox((0, 0), wrapped, font=font, stroke_width=stroke_width)
    text_w = tmp_bbox[2] - tmp_bbox[0]
    text_h = tmp_bbox[3] - tmp_bbox[1]

    x_range = max(1, bg.width - text_w - margin)
    y_range = max(1, bg.height - text_h - margin)
    x = random.randint(margin, x_range)
    y = random.randint(margin, y_range)

    fill = tuple(random.randint(0, 255) for _ in range(3))
    stroke_fill = tuple(random.randint(0, 255) for _ in range(3))

    draw.text(
        (x, y),
        wrapped,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    bbox_abs = draw.textbbox((x, y), wrapped, font=font, stroke_width=stroke_width)
    pad = stroke_width + 2
    bbox_abs = (bbox_abs[0] - pad, bbox_abs[1] - pad, bbox_abs[2] + pad, bbox_abs[3] + pad)
    bbox_abs = (
        max(0, bbox_abs[0]),
        max(0, bbox_abs[1]),
        min(bg.width, bbox_abs[2]),
        min(bg.height, bbox_abs[3]),
    )

    meta = {
        "font": str(font_path),
        "text": text,
        "wrapped_text": wrapped,
        "position": (x, y),
        "font_size": font_size,
        "fill": fill,
        "bbox": bbox_abs,
    }
    return bg, meta


def crop_bbox(image: Image.Image, bbox, padding: int = 4) -> Image.Image:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)
    return image.crop((x0, y0, x1, y1))


# ----------------------------
# Worker
# ----------------------------


def generate_family(
    family: str,
    font_paths: List[Path],
    render_candidates: List[Path],
    bg_cache_paths: List[Path],
    length_pool: List[int],
    text_pool: List[str],
    wiki_pool: List[str],
    target: int,
    out_root: Path,
    render_root: Path,
    json_root: Path,
    max_attempts: int,
    font_size_min: int,
    font_size_max: int,
    stroke_weights: List[int],
    stroke_weight_probs: List[float],
    padding: int,
    wiki_ratio: float,
) -> Dict:
    fam_dir = out_root / family
    fam_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(fam_dir.glob("*.jpg")))
    need = max(0, target - existing)
    if need == 0:
        return {"family": family, "generated": 0, "skipped_existing": existing, "fail": False}

    generated = 0
    attempts = 0
    last_err = None
    report_every = max(100, need // 10)
    while generated < need and attempts < max_attempts:
        attempts += 1
        try:
            background, _ = random_text_free_background(
                render_root,
                json_root,
                candidates=render_candidates,
                cache_paths=bg_cache_paths,
            )
            text = get_random_text(length_pool, text_pool, wiki_pool, wiki_ratio=wiki_ratio)
            font_size = random.randint(font_size_min, font_size_max)
            if stroke_weight_probs:
                stroke_width = random.choices(stroke_weights, weights=stroke_weight_probs, k=1)[0]
            else:
                stroke_width = random.choice(stroke_weights)
            img, meta = make_augmented_image(
                background,
                text,
                font_paths=font_paths,
                font_size=font_size,
                stroke_width=stroke_width,
                margin=24,
            )
            crop = crop_bbox(img, meta["bbox"], padding=padding)
            if crop.width < 2 or crop.height < 2:
                continue
            out_name = f"{uuid.uuid4().hex}.jpg"
            out_path = fam_dir / out_name
            crop.save(out_path, format="JPEG", quality=95, subsampling=1, optimize=False)
            generated += 1
            if generated % report_every == 0:
                print(f"[{family}] {generated}/{need} (attempts {attempts})")
        except Exception as exc:
            last_err = exc
            continue

    return {
        "family": family,
        "generated": generated,
        "skipped_existing": existing,
        "fail": generated < need,
        "need": need,
        "attempts": attempts,
        "last_err": repr(last_err) if last_err else "",
    }


# ----------------------------
# CLI
# ----------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Generate additional cropped font samples per family.")
    p.add_argument("--fonts_dir", type=Path, default=FONTS_DIR, help="Directory containing font files.")
    p.add_argument("--render_root", type=Path, default=RENDERED_ROOT, help="Rendered backgrounds root.")
    p.add_argument("--json_root", type=Path, default=JSON_ROOT, help="Layout JSON root.")
    p.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Output root.")
    p.add_argument("--target_per_family", type=int, default=5000, help="Number of new samples to create per family.")
    p.add_argument("--num_workers", type=int, default=os.cpu_count() or 4, help="Thread workers.")
    p.add_argument("--max_attempts_factor", type=float, default=6.0, help="Max attempts = target * factor.")
    p.add_argument("--wiki_ratio", type=float, default=0.5, help="Probability of using wiki text (if enabled).")
    p.add_argument("--use_wiki", action="store_true", help="Enable wiki text source. Default: off.")
    p.add_argument("--font_size_min", type=int, default=24, help="Min font size.")
    p.add_argument("--font_size_max", type=int, default=72, help="Max font size.")
    p.add_argument("--padding", type=int, default=8, help="Padding added around bbox before crop.")
    p.add_argument("--stroke_weights", type=int, nargs="+", default=[0, 1, 2], help="Stroke width candidates.")
    p.add_argument(
        "--stroke_weight_probs",
        type=float,
        nargs="+",
        default=[20.0, 2.0, 1.0],
        help="Stroke width sampling weights (same length as stroke_weights).",
    )
    p.add_argument("--use_bg_cache", action="store_true", default=True, help="Precache text-free backgrounds to local fast storage.")
    p.add_argument("--bg_cache_dir", type=Path, default=Path("/mnt/local/font_bg_cache"), help="Directory for cached backgrounds.")
    p.add_argument("--bg_cache_limit", type=int, default=50000, help="Max cached backgrounds when cache enabled.")
    return p.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    render_candidates = [d for d in args.render_root.iterdir() if d.is_dir()]
    if not render_candidates:
        raise RuntimeError(f"No render dirs found under {args.render_root}")

    font_families = build_font_families(args.fonts_dir)
    if not font_families:
        raise RuntimeError(f"No fonts found in {args.fonts_dir}")

    script_dir = Path(__file__).resolve().parent
    length_cache = script_dir / "text_length_pool.txt"
    text_cache = script_dir / "text_content_pool.txt"
    length_pool, _ = load_or_build_length_pool(args.json_root, length_cache, max_files=-1, max_chars=200)
    text_pool, _ = load_or_build_text_pool(args.json_root, text_cache, max_files=-1, max_chars=200)

    wiki_pool = []
    if args.use_wiki:
        wiki_pool = wiki_sentences(count=256, lang="en", max_chars=200, max_tries=50)
        print(f"Wiki pool: {len(wiki_pool)} sentences (enabled)")
    else:
        print("Wiki pool disabled; using JSON text only")

    families = sorted(font_families.keys())
    print(f"Families to generate: {len(families)} (target {args.target_per_family} each) using {len(render_candidates)} backgrounds")

    # Build / load background cache (optional)
    bg_cache_paths: List[Path] = []
    if args.use_bg_cache:
        try:
            args.bg_cache_dir.mkdir(parents=True, exist_ok=True)
            existing_cache = sorted(args.bg_cache_dir.glob("*.jpg"))
            if existing_cache:
                bg_cache_paths = existing_cache
                print(f"Using existing background cache: {len(bg_cache_paths)} images at {args.bg_cache_dir}")
            else:
                print(f"Building background cache to {args.bg_cache_dir} (limit {args.bg_cache_limit}) ...")
                cache_targets = list(render_candidates)
                random.shuffle(cache_targets)
                cache_targets = cache_targets[: args.bg_cache_limit]
                total_targets = len(cache_targets)
                if tqdm is not None:
                    progress = tqdm(total=total_targets, desc="bg_cache build", mininterval=1.0)
                else:
                    progress = None
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
                    futures = [
                        ex.submit(_cache_bg_one, sample_dir, args.json_root, args.bg_cache_dir, idx)
                        for idx, sample_dir in enumerate(cache_targets)
                    ]
                    for fut in concurrent.futures.as_completed(futures):
                        res = fut.result()
                        if res:
                            bg_cache_paths.append(res)
                        if progress:
                            progress.update(1)
                if progress:
                    progress.close()
                print(f"Cached {len(bg_cache_paths)} / {total_targets} backgrounds at {args.bg_cache_dir}")
        except Exception as exc:
            print(f"[warn] Failed to build/use bg cache ({exc}); falling back to on-the-fly backgrounds")
            bg_cache_paths = []

    results = []
    worker_args = {
        "length_pool": length_pool,
        "text_pool": text_pool,
        "wiki_pool": wiki_pool,
        "target": args.target_per_family,
        "out_root": args.output_dir,
        "render_root": args.render_root,
        "json_root": args.json_root,
        "render_candidates": render_candidates,
        "bg_cache_paths": bg_cache_paths,
        "max_attempts": int(args.target_per_family * args.max_attempts_factor),
        "font_size_min": args.font_size_min,
        "font_size_max": args.font_size_max,
        "stroke_weights": args.stroke_weights,
        "stroke_weight_probs": args.stroke_weight_probs,
        "padding": args.padding,
        "wiki_ratio": args.wiki_ratio,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        future_to_family = {
            ex.submit(
                generate_family,
                fam,
                font_families[fam],
                **worker_args,
            ): fam
            for fam in families
        }
        iterator = concurrent.futures.as_completed(future_to_family)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(families), desc="families")
        for fut in iterator:
            res = fut.result()
            results.append(res)
            fam = res["family"]
            if res.get("fail"):
                print(
                    f"[{fam}] generated {res.get('generated')} / {res.get('need')} "
                    f"(attempts {res.get('attempts')}, last_err={res.get('last_err')})"
                )
            else:
                print(f"[{fam}] generated {res.get('generated')} (existing {res.get('skipped_existing')})")

    total_new = sum(r.get("generated", 0) for r in results)
    failed = [r for r in results if r.get("fail")]
    print(f"\nDone. New images: {total_new}")
    if failed:
        print(f"Families that under-produced ({len(failed)}):")
        for r in failed[:20]:
            print(f"  {r['family']}: {r['generated']}/{r.get('need')} last_err={r.get('last_err')}")
        if len(failed) > 20:
            print("  ...")


if __name__ == "__main__":
    main()

