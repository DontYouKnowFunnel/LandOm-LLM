import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from html_tools.selector_lookup import map_funnel_items_to_selectors
from html_tools.spec import CompressionSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="funnel-json.json + input.html을 받아 selector 매핑 결과와 오버레이 JS를 생성합니다."
    )
    parser.add_argument("--input-html", default="input.html", help="원본 HTML 파일 경로")
    parser.add_argument("--funnel-json", default="funnel-json.json", help="퍼널 분석 JSON 파일 경로")
    parser.add_argument("--output-json", default="funnel_selector_output.json", help="selector 매핑 출력 JSON")
    parser.add_argument("--output-js", default="funnel_overlay.js", help="브라우저 콘솔 주입용 JS 파일")
    return parser.parse_args()


def build_overlay_js(mapped_items):
    data_json = json.dumps(mapped_items, ensure_ascii=False)
    return f"""(() => {{
  const DATA = {data_json};
  const ROOT_ID = "__funnel_overlay_root__";
  const STYLE_ID = "__funnel_overlay_style__";
  const funnelColors = {{
    HERO: "#ef4444",
    VALUE_PROP: "#f97316",
    FEATURE: "#eab308",
    SOCIAL_PROOF: "#22c55e",
    TRUST: "#06b6d4",
    PRICING: "#3b82f6",
    CTA: "#6366f1",
    INPUT_FORM: "#a855f7",
    CHECKOUT: "#ec4899",
    INTERACTIVE: "#14b8a6",
    FAQ: "#84cc16",
    POPUP: "#f43f5e"
  }};

  const oldRoot = document.getElementById(ROOT_ID);
  if (oldRoot) oldRoot.remove();
  const oldStyle = document.getElementById(STYLE_ID);
  if (oldStyle) oldStyle.remove();

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    #${{ROOT_ID}} {{
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 2147483647;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .funnel-box {{
      position: absolute;
      border: 2px solid;
      background: rgba(255,255,255,0.05);
      box-sizing: border-box;
      border-radius: 6px;
    }}
    .funnel-label {{
      position: absolute;
      top: -22px;
      left: 0;
      color: #fff;
      font-size: 11px;
      line-height: 1;
      padding: 4px 6px;
      border-radius: 6px;
      white-space: nowrap;
      box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }}
  `;
  document.head.appendChild(style);

  const root = document.createElement("div");
  root.id = ROOT_ID;
  document.body.appendChild(root);

  function render() {{
    root.innerHTML = "";
    DATA.forEach((item) => {{
      if (!item.selector) return;
      const el = document.querySelector(item.selector);
      if (!el) return;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      const color = funnelColors[item.funnel] || "#ffffff";
      const box = document.createElement("div");
      box.className = "funnel-box";
      box.style.borderColor = color;
      // Root is fixed to viewport, so use viewport coordinates directly.
      box.style.left = `${{rect.left}}px`;
      box.style.top = `${{rect.top}}px`;
      box.style.width = `${{rect.width}}px`;
      box.style.height = `${{rect.height}}px`;

      const label = document.createElement("div");
      label.className = "funnel-label";
      label.style.background = color;
      label.textContent = `${{item.funnel}} (${{Number(item.confidence || 0).toFixed(2)}})`;
      box.appendChild(label);
      root.appendChild(box);
    }});
  }}

  render();
  let rafId = null;
  const scheduleRender = () => {{
    if (rafId !== null) return;
    rafId = window.requestAnimationFrame(() => {{
      rafId = null;
      render();
    }});
  }};
  window.addEventListener("scroll", scheduleRender, {{ passive: true }});
  window.addEventListener("resize", scheduleRender);
  window.addEventListener("load", scheduleRender);
  console.log("[funnel-overlay] rendered");
}})();"""


def main() -> None:
    args = parse_args()
    spec = CompressionSpec()

    html = Path("input.html").read_text(encoding="utf-8")
    raw = json.loads(Path("funnel.json").read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise RuntimeError("funnel-json은 배열(JSON list) 형태여야 합니다.")

    mapped_items = map_funnel_items_to_selectors(
        html=html,
        funnel_items=raw,
        spec=spec,
        id_key="id",
    )

    matched = sum(1 for item in mapped_items if item.get("status") == "matched")
    result = {
        "summary": {
            "total": len(mapped_items),
            "matched": matched,
            "unmatched": len(mapped_items) - matched,
        },
        "items": mapped_items,
    }

    output_json_path = Path(args.output_json)
    output_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    overlay_js = build_overlay_js(mapped_items)
    output_js_path = Path(args.output_js)
    output_js_path.write_text(overlay_js, encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved JSON: {output_json_path}")
    print(f"Saved JS: {output_js_path}")


if __name__ == "__main__":
    main()
