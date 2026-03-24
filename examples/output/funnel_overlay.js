(() => {
  const DATA = [{"id": "eca26941", "funnel": "HERO", "confidence": 0.97, "selector": "section[id=\"hero\"]", "status": "matched"}, {"id": "ea66c06c", "funnel": "VALUE_PROP", "confidence": 0.93, "selector": "section[id=\"value-prop\"]", "status": "matched"}, {"id": "98c6f2c2", "funnel": "FEATURE", "confidence": 0.95, "selector": "section[id=\"features\"]", "status": "matched"}, {"id": "13cee27a", "funnel": "SOCIAL_PROOF", "confidence": 0.9, "selector": "section[id=\"social-proof\"]", "status": "matched"}, {"id": "3a170a9f", "funnel": "PRICING", "confidence": 0.98, "selector": "section[id=\"pricing\"]", "status": "matched"}, {"id": "12426c95", "funnel": "CTA", "confidence": 0.92, "selector": "section[id=\"cta\"]", "status": "matched"}, {"id": "449daf85", "funnel": "FAQ", "confidence": 0.96, "selector": "section[id=\"faq\"]", "status": "matched"}, {"id": "c81e728d", "funnel": "POPUP", "confidence": 0.99, "selector": "div[id=\"signup-modal\"]", "status": "matched"}];
  const ROOT_ID = "__funnel_overlay_root__";
  const STYLE_ID = "__funnel_overlay_style__";
  const funnelColors = {
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
  };

  const oldRoot = document.getElementById(ROOT_ID);
  if (oldRoot) oldRoot.remove();
  const oldStyle = document.getElementById(STYLE_ID);
  if (oldStyle) oldStyle.remove();

  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    #${ROOT_ID} {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 2147483647;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .funnel-box {
      position: absolute;
      border: 2px solid;
      background: rgba(255,255,255,0.05);
      box-sizing: border-box;
      border-radius: 6px;
    }
    .funnel-label {
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
    }
  `;
  document.head.appendChild(style);

  const root = document.createElement("div");
  root.id = ROOT_ID;
  document.body.appendChild(root);

  function render() {
    root.innerHTML = "";
    DATA.forEach((item) => {
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
      box.style.left = `${rect.left}px`;
      box.style.top = `${rect.top}px`;
      box.style.width = `${rect.width}px`;
      box.style.height = `${rect.height}px`;

      const label = document.createElement("div");
      label.className = "funnel-label";
      label.style.background = color;
      label.textContent = `${item.funnel} (${Number(item.confidence || 0).toFixed(2)})`;
      box.appendChild(label);
      root.appendChild(box);
    });
  }

  render();
  let rafId = null;
  const scheduleRender = () => {
    if (rafId !== null) return;
    rafId = window.requestAnimationFrame(() => {
      rafId = null;
      render();
    });
  };
  window.addEventListener("scroll", scheduleRender, { passive: true });
  window.addEventListener("resize", scheduleRender);
  window.addEventListener("load", scheduleRender);
  console.log("[funnel-overlay] rendered");
})();