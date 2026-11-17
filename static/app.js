// zchat front-end. Small on purpose: htmx does the page updates; this file handles
// streaming replies (EventSource), the composer box, and a few button helpers.

// ---------------------------------------------------------------- streaming replies

let activeStream = null; // { source: EventSource, mid: number }

// Any assistant bubble rendered with data-stream-url is waiting for a reply: connect to it.
function startPendingStreams(root) {
  root.querySelectorAll("[data-stream-url]:not([data-started])").forEach(startStream);
}

function startStream(bubble) {
  bubble.dataset.started = "1";
  const content = bubble.querySelector(".content");
  const reasoning = bubble.querySelector(".reasoning");
  const reasoningText = bubble.querySelector(".reasoning-text");
  const statusLine = bubble.querySelector(".status-line");
  let text = "";

  const source = new EventSource(bubble.dataset.streamUrl);
  activeStream = { source, mid: bubble.dataset.mid };
  setBusy(true);

  source.addEventListener("delta", (e) => {
    text += JSON.parse(e.data).t;
    if (content) content.textContent = text;
    scrollToBottom();
  });

  source.addEventListener("reasoning", (e) => {
    if (reasoning) {
      reasoning.hidden = false;
      reasoning.open = true;
    }
    if (reasoningText) reasoningText.textContent += JSON.parse(e.data).t;
    scrollToBottom();
  });

  source.addEventListener("status", (e) => {
    if (statusLine) statusLine.textContent = JSON.parse(e.data).t;
  });

  // The conversation got a title: refresh the sidebar list and the header.
  source.addEventListener("sidebar", (e) => {
    const data = JSON.parse(e.data);
    replaceElement(document.getElementById("conv-list"), data.html);
    const titleInput = document.querySelector(".title-input");
    if (titleInput) titleInput.value = data.title;
    document.title = data.title + " · zchat";
  });

  // Final event: swap in the fully rendered bubble (markdown, images, token counts).
  source.addEventListener("finished", (e) => {
    source.close();
    const data = JSON.parse(e.data);
    replaceElement(bubble, data.html);
    replaceElement(document.getElementById("chat-stats"), data.stats);
    replaceElement(document.getElementById("total-usage"), data.total);
    document.body.dispatchEvent(new Event("refresh-plan"));  // re-check the Coding Plan meters
    finishStream();
  });

  // Connection-level failure (server down, etc.). Without this, EventSource would retry forever.
  source.onerror = () => {
    source.close();
    if (statusLine) statusLine.textContent = "Connection lost. Reload the page to see what was saved.";
    finishStream();
  };
}

function finishStream() {
  activeStream = null;
  setBusy(false);
  scrollToBottom();
}

function stopStreaming() {
  if (!activeStream) return;
  // Ask the server to stop; it saves what it has so far and sends the "finished" event.
  fetch(`/stop?mid=${activeStream.mid}`, { method: "POST" });
}

function setBusy(busy) {
  document.body.classList.toggle("busy", busy);
}

// ---------------------------------------------------------------- DOM helpers

function replaceElement(oldEl, html) {
  if (!oldEl || !html) return;
  const holder = document.createElement("div");
  holder.innerHTML = html.trim();
  const newEl = holder.firstElementChild;
  oldEl.replaceWith(newEl);
  htmx.process(newEl);
}

function scrollToBottom(force) {
  const scroller = document.getElementById("scroller");
  if (!scroller) return;
  const nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 200;
  if (force || nearBottom) scroller.scrollTop = scroller.scrollHeight;
}

function toggleSidebar(open) {
  document.body.classList.toggle("sidebar-open", open);
}

function copyMessage(button) {
  const bubble = button.closest(".msg").querySelector(".content, .image-prompt p");
  if (!bubble) return;
  navigator.clipboard.writeText(bubble.innerText);
  button.classList.add("flash");
  setTimeout(() => button.classList.remove("flash"), 600);
}

// ---------------------------------------------------------------- composer

function toggleImageMode() {
  const modeInput = document.getElementById("mode-input");
  const input = document.getElementById("composer-input");
  const imageMode = modeInput.value !== "image";
  modeInput.value = imageMode ? "image" : "chat";
  document.getElementById("composer").classList.toggle("image-mode", imageMode);
  input.placeholder = imageMode ? "Describe the image to generate…" : "Message…  (Shift+Enter for a new line)";
  input.focus();
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 260) + "px";
}

function showAttachmentPreview() {
  const fileInput = document.getElementById("file-input");
  const preview = document.getElementById("attach-preview");
  if (!fileInput || !preview) return;
  preview.innerHTML = "";
  for (const file of fileInput.files) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${file.name} · ${Math.ceil(file.size / 1024)} KB`;
    preview.appendChild(chip);
  }
}

function resetComposer(form) {
  form.reset();
  const modeInput = document.getElementById("mode-input");
  if (modeInput) modeInput.value = form.classList.contains("image-mode") ? "image" : "chat";
  showAttachmentPreview();
  const input = document.getElementById("composer-input");
  if (input) autoResize(input);
}

document.addEventListener("keydown", (e) => {
  // Enter sends, Shift+Enter adds a new line.
  if (e.target.id === "composer-input" && e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!document.body.classList.contains("busy")) e.target.form.requestSubmit();
  }
});

document.addEventListener("input", (e) => {
  if (e.target.id === "composer-input") autoResize(e.target);
});

document.addEventListener("change", (e) => {
  if (e.target.id === "file-input") showAttachmentPreview();
});

// ---------------------------------------------------------------- htmx hooks

document.addEventListener("htmx:afterSwap", (e) => {
  startPendingStreams(document);
  if (e.detail.target.id === "messages") scrollToBottom(true);
});

document.addEventListener("htmx:afterRequest", (e) => {
  const form = e.detail.elt;
  if (form.id === "composer" && e.detail.successful) resetComposer(form);
});

// Don't send while a reply is still streaming (keeps the conversation tree tidy).
document.addEventListener("htmx:confirm", (e) => {
  if (e.detail.elt.id === "composer" && document.body.classList.contains("busy")) e.preventDefault();
});

document.addEventListener("DOMContentLoaded", () => {
  scrollToBottom(true);
  startPendingStreams(document);
  const input = document.getElementById("composer-input");
  if (input) input.focus();
});

// Light/dark toggle. The <head> script in main.py applies the saved choice on every page load.
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) {}  // private windows may block storage
}
