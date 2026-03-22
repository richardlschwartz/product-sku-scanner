#!/usr/bin/env python3
"""
Product SKU Scanner — Web App
Flask server that accepts an image upload, sends it to Claude Vision
for product/SKU identification, and displays results in the browser.
"""

import base64
import concurrent.futures
import json
import math
import os
import re
import statistics
import uuid
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import anthropic
import cv2

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_EXT = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT

VISION_PROMPT = """\
You are analyzing a product shelf or display photo. Identify every product position \
and estimate the count of INDIVIDUAL SELLABLE PACKAGES at each position.

Scan from TOP-LEFT to BOTTOM-RIGHT, row by row. A "row" is a horizontal shelf level.

=== ROW DETECTION ===

BEFORE identifying any products, first identify the ROWS (horizontal shelf levels) \
by looking for these physical landmarks:
  - SHELF DIVIDERS: horizontal surfaces (metal, wire, or wood shelves) that products sit on
  - PRICE TAGS / PRICE LABELS: small tags attached to the FRONT EDGE of shelf dividers. \
    Price tags are ALWAYS mounted on shelf edges, so a horizontal line of price tags = one \
    shelf divider = the boundary between two rows
  - SHELF EDGES: the visible front lip or rail of each shelf level

Count rows from TOP to BOTTOM:
  - Row 1 = the topmost shelf level (products sitting on the highest shelf divider)
  - Row 2 = the next shelf level down, and so on
  - Each horizontal shelf divider with products above it defines the BOTTOM of one row \
    and the TOP of the next
  - Products on the SAME physical shelf surface = same row, even if they are at \
    slightly different heights (e.g., tall vs short packages on the same shelf)
  - Do NOT split a single shelf into multiple rows just because products vary in height

=== POSITION ORDERING ===

Positions within each row MUST be numbered strictly LEFT to RIGHT based on their \
physical location in the image:
  - Position 1 = the leftmost product/bin/container in that row
  - Position 2 = the next one to the right, and so on
  - Use the physical containers (bins, crates, boxes, trays) as your guide — count \
    them left to right and assign position numbers in that order
  - For produce displays: each separate bin, crate, or compartment = one position. \
    Number them by their physical left-to-right order on the shelf
  - Do NOT group by product type — if lemons are in the leftmost bin, they are Position 1 \
    even if oranges appear in multiple bins to the right

=== WHAT TO COUNT ===

Count ONLY individual wrapped/packaged items that a customer could pick up and buy. \
Each physically separate package = 1 item.

CRITICAL — CARDBOARD DISPLAY BOXES AND TRAYS:
Most products on these shelves sit inside cardboard display boxes or trays. These boxes \
have a CUTOUT or CUTDOWN on the front — the front panel of the cardboard is cut lower \
so customers can reach inside. This creates a visible horizontal CUT LINE across the \
front of the box.

HOW TO COUNT ITEMS IN A DISPLAY BOX:
1. FIND THE CUT LINE — the horizontal edge where the front cardboard panel was cut down. \
   This is usually a rough or straight horizontal line across the front of the box, often \
   at roughly the midpoint of the box height.
2. EVERYTHING BELOW THE CUT LINE is the OUTSIDE of the box — this is printed cardboard \
   with brand logos, product photos, and advertising. It is FLAT. IGNORE IT COMPLETELY. \
   Nothing below the cut line is a real item.
3. ONLY LOOK ABOVE THE CUT LINE — through the opening, into the interior of the box. \
   What do you see inside?
   - If you see the flat cardboard BOTTOM of the box (plain brown/tan surface), or \
     mostly empty space with shadows → the box is EMPTY. Count = 0.
   - If you see actual packages sitting inside, count ONLY those physical packages \
     that are above the cut line and inside the box.
4. The area INSIDE the box is typically DARKER and more shadowed than the bright \
   printed exterior below the cut line. Use this contrast as a cue.

USE EDGE CONTRAST TO FIND THE CUT LINE:
The cut line creates a visible CONTRAST BOUNDARY across the front of the box:
  - BELOW the cut line: bright, colorful, evenly-lit printed graphics (the exterior)
  - ABOVE the cut line: darker, shadowed, uneven interior (or the back wall of the box)
Look for this brightness/color discontinuity. The printed exterior is typically BRIGHTER \
and more saturated than the interior. If the area above the cut line is dark, shadowed, \
or shows plain cardboard — the box is empty or nearly empty.

Also look for the RAW CARDBOARD EDGE itself — the cut line often shows the brown/tan \
inner layer of the cardboard where it was cut, creating a thin horizontal stripe that \
contrasts with the printed exterior below and the interior above.

COMMON MISTAKE TO AVOID:
A Reese's display box has bright orange Reese's product images PRINTED on its front \
panel (below the cut line). These are NOT real packages — they are flat ink on cardboard. \
The actual Reese's packages (if any) would be visible ABOVE the cut line, inside the box. \
If you only see the printed front panel and dark/shadowed space above the cut line, count = 0. \
The same applies to ALL display boxes — Kinder Bueno, Snickers, M&Ms, etc.

DO NOT COUNT:
  - Anything below the cut line on a display box — that is the printed exterior
  - The cardboard display container itself
  - Shelf labels, price tags, or signage
  - Printed product images on cardboard — these are advertising, not inventory

=== THREE COUNTING METHODS ===

Method A — Direct visual count:
  - Count each individual wrapped/packaged item you can see as a distinct physical object.
  - For items behind the front row, estimate based on visible depth and shadows.
  - Record as method_a_count.

Method B — Capacity-based estimate:
  - Estimate the space depth (rod length for pegs, shelf depth, or tray interior depth) \
    in inches. Record as space_depth_inches.
  - Estimate ONE package's depth in inches. Record as package_depth_inches.
  - Compute capacity = floor(space_depth_inches / package_depth_inches).
  - Estimate fullness (0-100%). Look at how much of the available space is occupied \
    by actual packages. Empty space = lower fullness. Record as method_b_fullness_pct.
  - Compute method_b_count = round(capacity * method_b_fullness_pct / 100).

Method C — Top-edge count (INDEPENDENT — do NOT anchor to Method A or B):
  - IMPORTANT: Perform this count BEFORE looking at your Method A or B results. \
    This must be an independent observation.
  - If the photo is taken at ANY angle where you can look even slightly down and see \
    the TOP EDGES of packages lined up front-to-back on a peg, shelf, or in a tray, \
    count those top edges carefully.
  - Each distinct top edge visible = one item. Count them one by one, front to back.
  - For peg-hung items: look along the peg rod from the front bag toward the back wall. \
    Each bag creates a visible top edge or ridge. Count every edge you can see, including \
    partially visible ones near the back. Standard slatwall peg rods are 10-12 inches long; \
    candy/snack bags are typically 1.5-2.5 inches thick, so expect 4-8 items per full peg.
  - If items are in multiple side-by-side columns, count top edges per column and multiply.
  - Set method_c_applicable = true if the viewing angle gives ANY usable top-down perspective \
    (even partial — most store photos are taken from a slightly elevated angle).
  - Set method_c_applicable = false ONLY if the photo is taken perfectly straight-on at \
    shelf level with zero downward angle.
  - Record the count as method_c_count.

Choose the final estimated_count:
  - Compare all applicable methods (A, B, and C if applicable).
  - If Method C is applicable and its count is HIGHER than A or B, prefer Method C — \
    top-edge counting reveals items hidden behind the front package that direct counting misses.
  - If two or more methods agree closely, prefer their consensus.
  - Only round DOWN when all methods agree or when Method C is not applicable.
  - When in doubt, choose the LOWER count.
  - If you can see 0 individual packages at a position, the count is 0. \
    An empty tray or shelf slot with no packages = 0.

=== END COUNTING METHODS ===

Return ONLY valid JSON (no markdown, no commentary):
{
  "rows": [
    {
      "row_number": 1,
      "positions": [
        {
          "position": 1,
          "product_name": "Brand Name Product Description Size",
          "display_type": "shelf",
          "method_a_count": 4,
          "space_depth_inches": 10,
          "package_depth_inches": 1.5,
          "method_b_capacity": 6,
          "method_b_fullness_pct": 50,
          "method_b_count": 3,
          "method_c_applicable": true,
          "method_c_count": 3,
          "chosen_method": "C",
          "estimated_count": 3,
          "counting_notes": "3 individually wrapped packages visible. Top edges confirm 3 items deep."
        }
      ]
    }
  ],
  "summary": {
    "total_rows": 3,
    "total_distinct_skus": 10,
    "total_items": 45
  }
}

Rules:
- Be specific with product names (brand, variant, size if readable).
- If you cannot read the label, describe visually (e.g., "Red can, unknown brand").
- display_type: "peg", "shelf", "bin", "tray", "hook", "slot".
- ALL numeric fields must be numbers, never null.
- method_c_applicable: true/false — can you see the top edges of packages from the camera angle?
- method_c_count: number of items counted by top edges (only when method_c_applicable = true, otherwise 0).
- chosen_method: "A", "B", "C", or a combination like "A+C".
- When in doubt, round DOWN.
- Remember: printed product images on cardboard are NOT inventory. Only count physical packages.
"""


VERIFY_PROMPT_TEMPLATE = """\
You previously analyzed this shelf photo and identified display box/tray positions. \
Now RECOUNT the items at each position listed below. Focus on ACCURACY.

For each position, find the display box/tray on the shelf and:
1. Find the CUT LINE — the horizontal edge where the front cardboard was cut down.
2. IGNORE everything BELOW the cut line — that is printed graphics on cardboard, NOT items.
3. Count ONLY physical packages you can see ABOVE the cut line, INSIDE the box.
4. If the interior above the cut line is DARK, SHADOWED, or shows empty cardboard → count = 0.
5. Printed product images on the box exterior are NOT real packages.

Positions to recount:
{positions_list}

Return ONLY valid JSON (no markdown):
{{
  "recounts": [
    {{"row": 1, "position": 1, "product_name": "...", "verified_count": 0, "reasoning": "what I see above the cut line"}}
  ]
}}
"""


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


MAX_BASE64_CHARS = 5_000_000  # Claude's 5 MB limit is on the base64 STRING, not decoded bytes


def encode_image(path: str) -> tuple:
    """Read, resize if needed, and base64-encode an image."""
    ext = os.path.splitext(path)[1].lower()
    img = Image.open(path)

    # Always use JPEG for encoding — much smaller for photos
    # Only keep PNG for images with alpha transparency
    if ext in (".png",) and img.mode == "RGBA":
        out_fmt, media_type = "PNG", "image/png"
    else:
        out_fmt, media_type = "JPEG", "image/jpeg"

    def _encode(image, fmt, quality=85):
        buf = BytesIO()
        if fmt == "JPEG":
            image = image.convert("RGB")  # JPEG can't handle alpha
            image.save(buf, format=fmt, quality=quality, optimize=True)
        else:
            image.save(buf, format=fmt, optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # Try encoding at original size first
    data = _encode(img, out_fmt)

    # If too large, progressively downscale (check base64 STRING length, not decoded bytes)
    scale = 1.0
    while len(data) > MAX_BASE64_CHARS and scale > 0.15:
        scale *= 0.75
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.LANCZOS)
        data = _encode(resized, out_fmt, quality=80)

    return data, media_type


def crop_region(path: str, bbox_pct: list) -> tuple:
    """Crop a region from the image using percentage-based coordinates.

    bbox_pct: [left%, top%, right%, bottom%] — each value 0-100.
    Returns (base64_data, media_type) of the cropped region.
    """
    img = Image.open(path)
    w, h = img.size

    # Convert percentages to pixels with 2% padding
    pad = 2
    left = max(0, int(w * (bbox_pct[0] - pad) / 100))
    top = max(0, int(h * (bbox_pct[1] - pad) / 100))
    right = min(w, int(w * (bbox_pct[2] + pad) / 100))
    bottom = min(h, int(h * (bbox_pct[3] + pad) / 100))

    # Ensure minimum crop size
    if right - left < 50 or bottom - top < 50:
        return None, None

    cropped = img.crop((left, top, right, bottom))

    # Encode cropped region
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        out_fmt, media_type = "JPEG", "image/jpeg"
    else:
        out_fmt, media_type = "PNG", "image/png"

    # Always use JPEG for crops (smaller)
    out_fmt, media_type = "JPEG", "image/jpeg"

    def _encode_crop(image, quality=85):
        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    data = _encode_crop(cropped)

    # Downscale if crop exceeds size limit (check base64 STRING length)
    scale = 1.0
    while len(data) > MAX_BASE64_CHARS and scale > 0.2:
        scale *= 0.75
        new_size = (int(cropped.width * scale), int(cropped.height * scale))
        resized = cropped.resize(new_size, Image.LANCZOS)
        data = _encode_crop(resized, quality=80)

    return data, media_type


def verify_tray_positions(client: anthropic.Anthropic, img_data: str, media_type: str,
                          positions_to_check: list) -> dict:
    """Send the full image again with a focused recount prompt for tray/box positions.

    Returns a dict of (row, position) → verified_count.
    """
    if not positions_to_check:
        return {}

    # Build the positions list for the prompt
    lines = []
    for pos_info in positions_to_check:
        row_num = pos_info["row"]
        pos_num = pos_info["position"]
        name = pos_info["product_name"]
        orig_count = pos_info["original_count"]
        lines.append(f"- Row {row_num}, Position {pos_num}: \"{name}\" (initial count: {orig_count})")

    positions_text = "\n".join(lines)
    prompt = VERIFY_PROMPT_TEMPLATE.replace("{positions_list}", positions_text)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
        parsed = json.loads(raw)

        # Build lookup
        result = {}
        for item in parsed.get("recounts", []):
            key = (item.get("row"), item.get("position"))
            result[key] = {
                "verified_count": item.get("verified_count", -1),
                "reasoning": item.get("reasoning", ""),
            }
        return result

    except Exception as e:
        print(f"Verification pass failed: {e}")
        return {}


def analyze_image(path: str) -> dict:
    """Send the image to Claude Vision and return parsed JSON results."""
    client = anthropic.Anthropic()
    img_data, media_type = encode_image(path)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16384,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    parsed = json.loads(raw)

    # Log raw response for debugging
    print("=== Claude Response ===")
    print(json.dumps(parsed, indent=2)[:3000])
    print("=== End Response ===")

    # ── Pass 1: Normalize fields ──
    for row in parsed.get("rows", []):
        for pos in row.get("positions", []):
            pos.setdefault("method_a_count", pos.get("estimated_count"))
            pos.setdefault("display_type", "shelf")
            pos.setdefault("space_depth_inches", pos.get("rod_length_inches"))
            pos.setdefault("package_depth_inches", None)
            pos.setdefault("method_b_capacity", None)
            pos.setdefault("method_b_fullness_pct", None)
            pos.setdefault("method_b_count", None)
            pos.setdefault("method_c_applicable", False)
            pos.setdefault("method_c_count", None)
            pos.setdefault("chosen_method", "A")
            pos.setdefault("counting_notes", "")
            pos.setdefault("verification_status", None)

    # Recalculate summary totals
    total_items = 0
    total_skus = 0
    row_count = 0
    for row in parsed.get("rows", []):
        row_count += 1
        for pos in row.get("positions", []):
            total_skus += 1
            total_items += pos.get("estimated_count", 0)

    parsed["summary"] = {
        "total_rows": row_count,
        "total_distinct_skus": total_skus,
        "total_items": total_items,
    }

    return parsed


def extract_frames(video_path: str, max_frames: int = 5, min_interval_sec: float = 2.0) -> list:
    """Extract evenly-spaced frames from a video file.

    Returns a list of dicts: [{"path": "/tmp/frame_0.jpg", "timestamp_sec": 0.0}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # Determine frame interval
    if duration_sec <= 0:
        raise ValueError("Video has zero duration")

    # Space frames evenly across the video, respecting min_interval
    num_frames = min(max_frames, max(1, int(duration_sec / min_interval_sec)))
    interval_sec = duration_sec / (num_frames + 1)  # +1 to avoid very start/end

    frames = []
    upload_dir = os.path.dirname(video_path)

    for i in range(num_frames):
        timestamp = interval_sec * (i + 1)
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_filename = f"{uuid.uuid4().hex}_frame_{i}.jpg"
        frame_path = os.path.join(upload_dir, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append({
            "path": frame_path,
            "filename": frame_filename,
            "timestamp_sec": round(timestamp, 1),
        })

    cap.release()
    return frames


def _normalize_name(name: str) -> str:
    """Normalize a product name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def aggregate_results(frame_results: list) -> dict:
    """Aggregate analysis results from multiple frames into a single result.

    Matches positions across frames by row_number + similar product name,
    then takes the median count for each position.
    """
    if not frame_results:
        return {"rows": [], "summary": {"total_rows": 0, "total_distinct_skus": 0, "total_items": 0}}

    if len(frame_results) == 1:
        r = frame_results[0]
        # Tag single-frame results
        for row in r.get("rows", []):
            for pos in row.get("positions", []):
                pos["frames_analyzed"] = 1
                pos["frame_counts"] = [pos.get("estimated_count", 0)]
        return r

    # Build a dict keyed by (row_number, normalized_name) → list of position dicts
    position_map = {}  # (row_num, norm_name) → [pos_dicts...]

    for result in frame_results:
        for row in result.get("rows", []):
            row_num = row.get("row_number", 0)
            for pos in row.get("positions", []):
                name = pos.get("product_name", "")
                norm = _normalize_name(name)
                key = (row_num, norm)

                # Try exact key match first; if not, check for partial overlap
                matched_key = None
                for existing_key in position_map:
                    if existing_key[0] == row_num:
                        # Check if names share at least 60% of characters
                        existing_norm = existing_key[1]
                        if norm and existing_norm:
                            overlap = sum(1 for c in norm if c in existing_norm)
                            similarity = overlap / max(len(norm), len(existing_norm))
                            if similarity > 0.6:
                                matched_key = existing_key
                                break

                if matched_key:
                    position_map[matched_key].append(pos)
                else:
                    position_map[key] = [pos]

    # Now build aggregated rows
    rows_dict = {}  # row_num → list of aggregated positions
    for (row_num, _), positions in position_map.items():
        counts = [p.get("estimated_count", 0) for p in positions]
        median_count = int(round(statistics.median(counts)))

        # For box emptiness: majority vote
        empty_votes = sum(1 for p in positions if p.get("box_appears_empty", False))
        majority_empty = empty_votes > len(positions) / 2

        # Pick the "best" frame's data as the base (highest confidence / most detail)
        base = max(positions, key=lambda p: len(p.get("counting_notes", "")))
        base = dict(base)  # copy

        base["estimated_count"] = median_count
        base["frames_analyzed"] = len(positions)
        base["frame_counts"] = counts

        if majority_empty:
            base["box_appears_empty"] = True
            base["box_fill_assessment"] = "empty_box"
            base["estimated_count"] = 0

        # Collect per-method counts across frames for transparency
        a_counts = [p.get("method_a_count", 0) for p in positions if p.get("method_a_count") is not None]
        b_counts = [p.get("method_b_count", 0) for p in positions if p.get("method_b_count") is not None]
        c_counts = [p.get("method_c_count", 0) for p in positions if p.get("method_c_count") is not None]
        if a_counts:
            base["method_a_count"] = int(round(statistics.median(a_counts)))
        if b_counts:
            base["method_b_count"] = int(round(statistics.median(b_counts)))
        if c_counts:
            base["method_c_count"] = int(round(statistics.median(c_counts)))

        rows_dict.setdefault(row_num, []).append(base)

    # Build final structure
    rows = []
    for row_num in sorted(rows_dict.keys()):
        positions = sorted(rows_dict[row_num], key=lambda p: p.get("position", 0))
        rows.append({"row_number": row_num, "positions": positions})

    total_items = sum(p.get("estimated_count", 0) for r in rows for p in r["positions"])
    total_skus = sum(len(r["positions"]) for r in rows)

    return {
        "rows": rows,
        "frames_analyzed": len(frame_results),
        "summary": {
            "total_rows": len(rows),
            "total_distinct_skus": total_skus,
            "total_items": total_items,
        },
    }


def analyze_video(video_path: str) -> dict:
    """Extract frames from a video, analyze each, and aggregate results."""
    frames = extract_frames(video_path, max_frames=5)
    print(f"=== Extracted {len(frames)} frames from video ===")

    frame_results = []
    frame_thumbnails = []

    for i, frame_info in enumerate(frames):
        print(f"=== Analyzing frame {i + 1}/{len(frames)} (t={frame_info['timestamp_sec']}s) ===")
        try:
            result = analyze_image(frame_info["path"])
            frame_results.append(result)
            frame_thumbnails.append({
                "url": f"/uploads/{frame_info['filename']}",
                "timestamp_sec": frame_info["timestamp_sec"],
            })
        except Exception as e:
            print(f"  Frame {i + 1} analysis failed: {e}")
            continue

    if not frame_results:
        raise ValueError("No frames could be analyzed from the video")

    aggregated = aggregate_results(frame_results)
    aggregated["frame_thumbnails"] = frame_thumbnails
    aggregated["frames_analyzed"] = len(frame_results)
    return aggregated


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(sorted(ALLOWED_EXT))}"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        is_video = ext in ALLOWED_VIDEO_EXT
        if is_video:
            data = analyze_video(filepath)
            # Use first frame thumbnail as the display image
            first_thumb = data.get("frame_thumbnails", [{}])[0].get("url", "")
            return jsonify({"image_url": first_thumb, "is_video": True, "results": data})
        else:
            data = analyze_image(filepath)
            return jsonify({"image_url": f"/uploads/{filename}", "is_video": False, "results": data})
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse Claude response: {e}"}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"Claude API error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Product SKU Scanner at http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
