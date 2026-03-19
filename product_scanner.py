#!/usr/bin/env python3
"""
Product SKU Scanner
Opens a photo, sends it to Claude Vision to identify products and counts,
and displays the results in a GUI window organized by position (row by row).
"""

import base64
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from io import BytesIO

from PIL import Image, ImageTk
import anthropic


SUPPORTED_FORMATS = [
    ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tiff *.tif"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("GIF", "*.gif"),
    ("BMP", "*.bmp"),
    ("WebP", "*.webp"),
    ("TIFF", "*.tiff *.tif"),
    ("All files", "*.*"),
]

VISION_PROMPT = """\
You are analyzing a product shelf or display photo. Your job is to identify every \
distinct product SKU visible and estimate the count of each at every position.

Scan the image systematically from TOP-LEFT to BOTTOM-RIGHT, row by row. \
A "row" is a horizontal band of products at roughly the same vertical level \
(like a shelf row). Within each row, list products from left to right.

Return ONLY valid JSON with this exact structure (no markdown, no commentary):
{
  "rows": [
    {
      "row_number": 1,
      "positions": [
        {
          "position": 1,
          "product_name": "Brand Name Product Description Size",
          "estimated_count": 3
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
- Be as specific as possible with product names (include brand, variant, size if readable).
- If you cannot read the label, describe the product visually (e.g., "Red can, unknown brand").
- estimated_count is your best guess for how many units of that SKU are at that position \
(including items partially hidden behind front-facing ones).
- Each position represents a distinct facing/group of the same SKU on the shelf.
- If the image does not contain products or shelves, still return the JSON structure \
with an empty rows array and zeros in summary.
"""


def encode_image(path: str) -> tuple[str, str]:
    """Read and base64-encode an image, returning (base64_data, media_type)."""
    ext = os.path.splitext(path)[1].lower()
    media_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/png",   # convert bmp to png for API
        ".webp": "image/webp",
        ".tiff": "image/png",  # convert tiff to png for API
        ".tif": "image/png",
    }
    media_type = media_map.get(ext, "image/png")

    # For formats the API may not accept natively, convert via PIL
    if ext in (".bmp", ".tiff", ".tif"):
        img = Image.open(path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        media_type = "image/png"
    else:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def analyze_image(path: str) -> dict:
    """Send the image to Claude and get product analysis back."""
    client = anthropic.Anthropic()
    img_data, media_type = encode_image(path)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
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
                    {
                        "type": "text",
                        "text": VISION_PROMPT,
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return json.loads(raw)


class ProductScannerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Product SKU Scanner")
        self.root.geometry("1100x750")
        self.root.minsize(800, 500)

        self._build_ui()

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Open Image…", command=self._open_file).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Select an image to scan.")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        # Paned window: image on left, results on right
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Image panel
        img_frame = ttk.LabelFrame(pane, text="Photo", padding=4)
        pane.add(img_frame, weight=1)

        self.img_canvas = tk.Canvas(img_frame, bg="#2b2b2b")
        self.img_canvas.pack(fill=tk.BOTH, expand=True)
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)
        self._pil_image = None
        self._tk_image = None

        # Results panel
        res_frame = ttk.LabelFrame(pane, text="Product Analysis", padding=4)
        pane.add(res_frame, weight=1)

        self.results_text = tk.Text(
            res_frame, wrap=tk.WORD, font=("Menlo", 12), state=tk.DISABLED,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4",
            padx=8, pady=8,
        )
        scrollbar = ttk.Scrollbar(res_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for formatting
        self.results_text.tag_configure("heading", font=("Menlo", 14, "bold"), foreground="#569cd6")
        self.results_text.tag_configure("subheading", font=("Menlo", 12, "bold"), foreground="#4ec9b0")
        self.results_text.tag_configure("product", font=("Menlo", 12), foreground="#ce9178")
        self.results_text.tag_configure("count", font=("Menlo", 12, "bold"), foreground="#b5cea8")
        self.results_text.tag_configure("summary", font=("Menlo", 12, "italic"), foreground="#dcdcaa")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select a product image",
            filetypes=SUPPORTED_FORMATS,
        )
        if not path:
            return

        # Display the image
        try:
            self._pil_image = Image.open(path)
            self._display_image()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            return

        # Analyze in background thread
        self.status_var.set("Analyzing image with Claude Vision…")
        self._set_results("Analyzing… please wait.\n")

        def _run():
            try:
                data = analyze_image(path)
                self.root.after(0, lambda: self._show_results(data))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _display_image(self):
        if self._pil_image is None:
            return
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        img = self._pil_image.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(img)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw // 2, ch // 2, image=self._tk_image, anchor=tk.CENTER)

    def _on_canvas_resize(self, _event):
        self._display_image()

    def _set_results(self, text: str):
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.configure(state=tk.DISABLED)

    def _show_results(self, data: dict):
        self.status_var.set("Analysis complete.")
        t = self.results_text
        t.configure(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        rows = data.get("rows", [])
        summary = data.get("summary", {})

        if not rows:
            t.insert(tk.END, "No products detected in this image.\n", "summary")
            t.configure(state=tk.DISABLED)
            return

        t.insert(tk.END, "PRODUCT SCAN RESULTS\n", "heading")
        t.insert(tk.END, "=" * 40 + "\n\n")

        for row in rows:
            row_num = row.get("row_number", "?")
            t.insert(tk.END, f"Row {row_num}\n", "subheading")
            t.insert(tk.END, "-" * 30 + "\n")

            for pos in row.get("positions", []):
                name = pos.get("product_name", "Unknown")
                count = pos.get("estimated_count", "?")
                pos_num = pos.get("position", "?")
                t.insert(tk.END, f"  Pos {pos_num}: ", "count")
                t.insert(tk.END, f"{name}", "product")
                t.insert(tk.END, f"  ×{count}\n", "count")

            t.insert(tk.END, "\n")

        t.insert(tk.END, "=" * 40 + "\n", "heading")
        t.insert(tk.END, "SUMMARY\n", "heading")
        t.insert(tk.END, f"  Rows: {summary.get('total_rows', '?')}\n", "summary")
        t.insert(tk.END, f"  Distinct SKUs: {summary.get('total_distinct_skus', '?')}\n", "summary")
        t.insert(tk.END, f"  Total items: {summary.get('total_items', '?')}\n", "summary")

        t.configure(state=tk.DISABLED)

    def _show_error(self, msg: str):
        self.status_var.set("Error during analysis.")
        self._set_results(f"Error:\n{msg}\n\nMake sure ANTHROPIC_API_KEY is set.")
        messagebox.showerror("Analysis Error", msg)


def main():
    root = tk.Tk()
    ProductScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
