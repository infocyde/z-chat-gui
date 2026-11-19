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
    if (statusLine) statusLine.textContent = "";  // e.g. "Summarizing the story so far…" is over
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

  // Older messages were just summarized: show the "Story so far" box after the message it covers.
  source.addEventListener("summary", (e) => {
    const data = JSON.parse(e.data);
    document.querySelectorAll(".story-summary").forEach((el) => el.remove());  // only the newest shows
    const covered = document.getElementById("msg-" + data.mid);
    if (covered) {
      covered.insertAdjacentHTML("afterend", data.html);
      htmx.process(covered.nextElementSibling);  // wire up its Edit / Remove buttons
    }
  });

  // The lorebook just updated itself: show the note after the reply it's tied to, refresh the drawer.
  source.addEventListener("lore", (e) => {
    const data = JSON.parse(e.data);
    const reply = document.getElementById("msg-" + data.mid);
    if (reply) {
      reply.insertAdjacentHTML("afterend", data.html);
      htmx.process(reply.nextElementSibling);  // wire up its Undo button
    }
    replaceElement(document.getElementById("chat-lore"), data.drawer);
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

// ---------------------------------------------------------------- long chats: newest messages only

// A long chat opens with only its newest messages (the server sends RECENT_MESSAGES of them, see main.py).
// Scrolling to the top or pressing the up arrow loads the rest. The next message you send trims it back.
const RECENT_MESSAGES = 10;  // keep in step with RECENT_MESSAGES in main.py

let loadingOlder = false;

// Load every message before the first one on screen, in place of the "Show earlier messages" button.
async function showOlderMessages(goToTop) {
  const button = document.getElementById("older-messages");
  const scroller = document.getElementById("scroller");
  if (button && !loadingOlder) {
    loadingOlder = true;
    const firstMessage = document.querySelector("#messages > .msg");
    const fromBottom = scroller.scrollHeight - scroller.scrollTop;
    try {
      await htmx.ajax("GET", `/older?before=${firstMessage.dataset.mid}`, { target: button, swap: "outerHTML" });
      // The new messages went in above you: stay on the message you were reading.
      scroller.scrollTop = scroller.scrollHeight - fromBottom;
    } finally {
      loadingOlder = false;
    }
  }
  if (goToTop) scroller.scrollTo({ top: 0, behavior: "smooth" });
}

// After you send, a chat that has its older messages loaded goes back to only the newest ones.
function trimToRecent() {
  const list = document.getElementById("messages");
  const messages = list.querySelectorAll(":scope > .msg");
  if (messages.length <= RECENT_MESSAGES) return;
  // Remove everything above the first message we keep, including the notes and summary between them.
  const firstKept = messages[messages.length - RECENT_MESSAGES];
  while (list.firstElementChild !== firstKept) list.firstElementChild.remove();
  // The same button ui.older_messages_button() draws.
  const button = document.createElement("button");
  button.type = "button";
  button.id = "older-messages";
  button.className = "btn btn-ghost older-btn";
  button.textContent = "Show earlier messages";
  button.onclick = () => showOlderMessages(false);
  list.prepend(button);
}

function jumpToTop() {
  showOlderMessages(true);
}

function jumpToBottom() {
  const scroller = document.getElementById("scroller");
  scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
}

let lastScrollTop = 0;

// Scroll events don't bubble, so listen in the capture phase to catch the chat's scroller.
document.addEventListener("scroll", (e) => {
  if (e.target.id !== "scroller") return;
  const scroller = e.target;
  // Only when you scroll up to the top yourself, not when the page jumps to the bottom on load.
  const scrollingUp = scroller.scrollTop < lastScrollTop;
  lastScrollTop = scroller.scrollTop;
  if (scrollingUp && scroller.scrollTop < 150) showOlderMessages(false);
  // Show the up / down arrows only once you've scrolled away from the newest message.
  const fromBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
  document.getElementById("jump-btns")?.classList.toggle("show", fromBottom > 300);
}, true);

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

// Clear only the message box and attachments. Not form.reset(): on the new-chat page the same form holds
// the system prompt, model and Lorekeeper switch, and a warning like "Type a message" must not wipe them.
function resetComposer() {
  const input = document.getElementById("composer-input");
  if (input) {
    input.value = "";
    autoResize(input);
  }
  const fileInput = document.getElementById("file-input");
  if (fileInput) fileInput.value = "";
  showAttachmentPreview();
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

// ---------------------------------------------------------------- scenario categories

// The categories box holds "Fantasy, Mystery". Clicking a suggestion button adds or removes that name.
function categoryList(input) {
  return input.value.split(",").map((c) => c.trim()).filter((c) => c);
}

function toggleCategory(button) {
  const input = document.getElementById("categories-input");
  const name = button.textContent;
  const list = categoryList(input);
  const kept = list.filter((c) => c.toLowerCase() !== name.toLowerCase());
  if (kept.length === list.length) kept.push(name); // it wasn't there, so add it
  input.value = kept.join(", ");
  highlightCategoryButtons();
}

// Light up the buttons whose category is in the box (also while typing in it).
function highlightCategoryButtons() {
  const input = document.getElementById("categories-input");
  if (!input) return;
  const chosen = categoryList(input).map((c) => c.toLowerCase());
  document.querySelectorAll(".tag-btn").forEach((b) => {
    b.classList.toggle("on", chosen.includes(b.textContent.toLowerCase()));
  });
}

document.addEventListener("input", (e) => {
  if (e.target.id === "categories-input") highlightCategoryButtons();
});

// ---------------------------------------------------------------- htmx hooks

document.addEventListener("htmx:afterSwap", (e) => {
  startPendingStreams(document);
  if (e.detail.target.id === "messages") {
    trimToRecent();  // a send adds to the list; keep only the newest messages on the page
    scrollToBottom(true);
  }
});

document.addEventListener("htmx:afterRequest", (e) => {
  const form = e.detail.elt;
  if (form.id === "composer" && e.detail.successful) resetComposer();
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
