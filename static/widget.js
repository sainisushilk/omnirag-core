/*!
 * OmniRAG Core chat widget.
 * Usage: <script src="https://your-server/static/widget.js" defer></script>
 */
(function () {
  "use strict";

  if (window.__omniragWidgetLoaded) return;
  window.__omniragWidgetLoaded = true;

  var SCRIPT_PATH = /\/static\/widget\.js(?:[?#].*)?$/;

  // document.currentScript is only defined while the script executes, so it is
  // read immediately. Script optimizers may inject the tag asynchronously, in
  // which case the matching tag is located in the page.
  var script = document.currentScript;
  if (!script || !script.src) {
    var candidates = document.querySelectorAll("script[src]");
    for (var i = candidates.length - 1; i >= 0; i--) {
      if (SCRIPT_PATH.test(candidates[i].src)) {
        script = candidates[i];
        break;
      }
    }
  }
  if (!script || !script.src) {
    console.error("[OmniRAG] The widget could not determine its server URL.");
    return;
  }
  // Keeps any sub-path, so https://host/rag/static/widget.js maps to https://host/rag.
  var apiBase = SCRIPT_PATH.test(script.src)
    ? script.src.replace(SCRIPT_PATH, "")
    : new URL(script.src).origin;

  var MAX_QUESTION_LENGTH = 2000;
  var REQUEST_TIMEOUT_MS = 120000;
  var config = { bot_title: "AI Assistant", brand_color: "#4f46e5" };
  var ui = {};
  var isOpen = false;
  var isBusy = false;

  function css(color, textColor) {
    return [
      "#omnirag-root{all:initial!important;}",
      "#omnirag-root *:not(path){all:unset;box-sizing:border-box;}",
      "#omnirag-root path{color:inherit;}",
      "#omnirag-root *:not(path){font-family:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}",
      "#omnirag-root #omnirag-launcher{position:fixed;right:20px;bottom:20px;z-index:2147483646;width:56px;height:56px;border-radius:50%;background:" + color + ";color:" + textColor + ";display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.25);transition:transform .15s ease;}",
      "#omnirag-root #omnirag-launcher:hover{transform:scale(1.06);}",
      "#omnirag-root #omnirag-launcher svg,#omnirag-root #omnirag-close svg,#omnirag-root #omnirag-send svg{display:block;width:24px;height:24px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;pointer-events:none;}",
      "#omnirag-root #omnirag-launcher path,#omnirag-root #omnirag-close path,#omnirag-root #omnirag-send path{fill:none;stroke:currentColor;stroke-width:2;}",
      "#omnirag-root #omnirag-panel{position:fixed;right:20px;bottom:90px;z-index:2147483647;width:370px;height:540px;max-width:calc(100vw - 40px);max-height:calc(100vh - 110px);display:none;flex-direction:column;overflow:hidden;background:#fff;border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.28);}",
      "#omnirag-root #omnirag-panel.omnirag-open{display:flex;}",
      "#omnirag-root #omnirag-header{display:flex;flex:none;align-items:center;justify-content:space-between;gap:8px;padding:14px 16px;background:" + color + ";color:" + textColor + ";}",
      "#omnirag-root #omnirag-title{display:block;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:16px;font-weight:600;line-height:1.3;color:inherit;}",
      "#omnirag-root #omnirag-close{display:flex;flex:none;padding:4px;border-radius:6px;cursor:pointer;color:inherit;}",
      "#omnirag-root #omnirag-close svg{width:20px;height:20px;}",
      "#omnirag-root #omnirag-messages{display:flex;flex:1;min-height:0;flex-direction:column;gap:10px;padding:16px;overflow-y:auto;overscroll-behavior:contain;background:#f6f7f9;}",
      "#omnirag-root .omnirag-message{display:block;flex:none;max-width:85%;padding:10px 13px;border-radius:14px;font-size:14px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;cursor:text;user-select:text;-webkit-user-select:text;}",
      "#omnirag-root .omnirag-bot{align-self:flex-start;background:#fff;color:#1f2937;border:1px solid #e5e7eb;border-bottom-left-radius:4px;}",
      "#omnirag-root .omnirag-user{align-self:flex-end;background:" + color + ";color:" + textColor + ";border-bottom-right-radius:4px;}",
      "#omnirag-root .omnirag-error{background:#fef2f2;color:#991b1b;border-color:#fecaca;}",
      "#omnirag-root .omnirag-sources{display:block;margin-top:8px;padding-top:7px;border-top:1px solid #e5e7eb;font-size:12px;line-height:1.5;color:#4b5563;white-space:normal;}",
      "#omnirag-root .omnirag-source{display:block;font-size:12px;color:#4b5563;overflow-wrap:anywhere;}",
      "#omnirag-root a.omnirag-source{color:#1d4ed8;text-decoration:underline;cursor:pointer;}",
      "#omnirag-root #omnirag-form{display:flex;flex:none;align-items:flex-end;gap:8px;padding:12px;background:#fff;border-top:1px solid #e5e7eb;}",
      "#omnirag-root #omnirag-input{display:block;flex:1;min-width:0;height:42px;max-height:110px;padding:10px 12px;border:1px solid #9ca3af;border-radius:10px;background:#fff;color:#1f2937;font-size:16px;line-height:20px;white-space:pre-wrap;overflow-wrap:anywhere;overflow-y:auto;resize:none;cursor:text;}",
      "#omnirag-root #omnirag-input::placeholder{color:#6b7280;opacity:1;}",
      "#omnirag-root #omnirag-send{display:flex;flex:none;width:42px;height:42px;align-items:center;justify-content:center;border-radius:10px;background:" + color + ";color:" + textColor + ";cursor:pointer;}",
      "#omnirag-root #omnirag-send svg{width:20px;height:20px;}",
      "#omnirag-root #omnirag-send:disabled{opacity:.5;cursor:default;}",
      "#omnirag-root #omnirag-launcher:focus-visible,#omnirag-root #omnirag-input:focus-visible,#omnirag-root #omnirag-send:focus-visible,#omnirag-root #omnirag-close:focus-visible,#omnirag-root a.omnirag-source:focus-visible{outline:3px solid #93c5fd;outline-offset:1px;}",
      "@media (max-width:480px){#omnirag-root #omnirag-panel{right:0;bottom:0;left:0;top:0;width:100%;height:100%;height:100dvh;max-width:none;max-height:none;border-radius:0;}}"
    ].join("\n");
  }

  function readableTextColor(hex) {
    var channels = [1, 3, 5].map(function (offset) {
      var value = parseInt(hex.substr(offset, 2), 16) / 255;
      return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
    });
    var luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
    return luminance > 0.36 ? "#111827" : "#ffffff";
  }

  function element(tag, attributes, children) {
    var node = document.createElement(tag);
    Object.keys(attributes || {}).forEach(function (name) {
      node.setAttribute(name, attributes[name]);
    });
    (children || []).forEach(function (child) {
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  function icon(pathData) {
    var namespace = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(namespace, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    var path = document.createElementNS(namespace, "path");
    path.setAttribute("d", pathData);
    svg.appendChild(path);
    return svg;
  }

  function applyTheme() {
    var color = /^#[0-9a-f]{6}$/i.test(config.brand_color) ? config.brand_color : "#4f46e5";
    ui.style.textContent = css(color, readableTextColor(color));
    ui.title.textContent = config.bot_title;
    ui.panel.setAttribute("aria-label", config.bot_title);
  }

  function addMessage(text, kind) {
    var message = element("div", { class: "omnirag-message " + kind }, [text]);
    ui.messages.appendChild(message);
    ui.messages.scrollTop = ui.messages.scrollHeight;
    return message;
  }

  function addSources(message, sources) {
    if (!Array.isArray(sources) || !sources.length) return;
    var list = element("div", { class: "omnirag-sources" }, ["Sources"]);
    sources.forEach(function (source) {
      source = String(source);
      list.appendChild(
        /^https?:\/\//i.test(source)
          ? element("a", { class: "omnirag-source", href: source, target: "_blank", rel: "noopener noreferrer" }, [source])
          : element("span", { class: "omnirag-source" }, [source])
      );
    });
    message.appendChild(list);
    ui.messages.scrollTop = ui.messages.scrollHeight;
  }

  function setBusy(busy) {
    isBusy = busy;
    ui.send.disabled = busy;
    ui.messages.setAttribute("aria-busy", String(busy));
  }

  function toggle(open) {
    isOpen = open;
    ui.panel.classList.toggle("omnirag-open", open);
    ui.launcher.setAttribute("aria-expanded", String(open));
    ui.launcher.setAttribute("aria-label", open ? "Close chat" : "Open chat");
    if (open) {
      if (!ui.messages.childNodes.length) addMessage("Hi! How can I help you today?", "omnirag-bot");
      ui.input.focus();
    } else {
      ui.launcher.focus();
    }
  }

  function resizeInput() {
    ui.input.style.height = "42px";
    ui.input.style.height = Math.min(ui.input.scrollHeight + 2, 110) + "px";
  }

  function ask(event) {
    if (event) event.preventDefault();
    var question = ui.input.value.trim();
    if (!question || isBusy) return;
    ui.input.value = "";
    resizeInput();
    addMessage(question, "omnirag-user");
    var pending = addMessage("Thinking…", "omnirag-bot");
    setBusy(true);

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS);
    fetch(apiBase + "/api/public/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "omit",
      signal: controller.signal,
      body: JSON.stringify({ question: question })
    })
      .then(function (response) {
        return response.json().catch(function () { return {}; }).then(function (data) {
          if (!response.ok || typeof data.answer !== "string") {
            var failure = new Error("chat request failed");
            failure.detail = typeof data.detail === "string" ? data.detail : "";
            throw failure;
          }
          pending.textContent = data.answer;
          addSources(pending, data.sources);
        });
      })
      .catch(function (error) {
        pending.className += " omnirag-error";
        pending.textContent = error.detail || "Sorry, I could not reach the assistant. Please try again.";
      })
      .then(function () {
        clearTimeout(timer);
        setBusy(false);
        ui.messages.scrollTop = ui.messages.scrollHeight;
        if (isOpen) ui.input.focus();
      });
  }

  function build() {
    ui.style = element("style", { id: "omnirag-style" });
    ui.title = element("div", { id: "omnirag-title" });
    ui.close = element("button", { id: "omnirag-close", type: "button", "aria-label": "Close chat" }, [icon("M6 6l12 12M18 6L6 18")]);
    ui.messages = element("div", { id: "omnirag-messages", role: "log", "aria-live": "polite" });
    ui.input = element("textarea", {
      id: "omnirag-input", rows: "1", maxlength: String(MAX_QUESTION_LENGTH),
      placeholder: "Type your question", "aria-label": "Type your question"
    });
    ui.send = element("button", { id: "omnirag-send", type: "submit", "aria-label": "Send" }, [icon("M5 12h14M13 6l6 6-6 6")]);
    ui.form = element("form", { id: "omnirag-form" }, [ui.input, ui.send]);
    ui.panel = element("div", { id: "omnirag-panel", role: "dialog" }, [
      element("div", { id: "omnirag-header" }, [ui.title, ui.close]),
      ui.messages,
      ui.form
    ]);
    ui.launcher = element("button", {
      id: "omnirag-launcher", type: "button", "aria-label": "Open chat",
      "aria-expanded": "false", "aria-controls": "omnirag-panel"
    }, [icon("M21 12a8 8 0 0 1-8 8H4l2-3.2A8 8 0 1 1 21 12z")]);

    applyTheme();
    document.head.appendChild(ui.style);
    document.body.appendChild(element("div", { id: "omnirag-root" }, [ui.launcher, ui.panel]));

    ui.launcher.addEventListener("click", function () { toggle(!isOpen); });
    ui.close.addEventListener("click", function () { toggle(false); });
    ui.form.addEventListener("submit", ask);
    ui.input.addEventListener("input", resizeInput);
    // Host-page keyboard shortcuts must not react to typing in the chat.
    ui.panel.addEventListener("keydown", function (event) {
      event.stopPropagation();
      if (event.key === "Escape") toggle(false);
    });
    ui.input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) ask(event);
    });
  }

  function start() {
    build();
    fetch(apiBase + "/api/public/widget-config", { credentials: "omit" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        if (data.bot_title) config.bot_title = String(data.bot_title);
        if (data.brand_color) config.brand_color = String(data.brand_color);
        applyTheme();
      })
      .catch(function (error) {
        console.warn("[OmniRAG] Using the default appearance:", error.message);
      });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
