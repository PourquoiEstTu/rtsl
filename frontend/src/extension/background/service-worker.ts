let translatorTabId: number | null = null;

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "START_TRANSLATOR") {
    handleStartTranslator(message)
      .then(() => sendResponse({ ok: true }))
      .catch((error: unknown) => {
        console.error("START_TRANSLATOR error:", error);
        sendResponse({
          ok: false,
          error: error instanceof Error ? error.message : String(error)
        });
      });
    return true;
  }

  if (message.type === "STOP_TRANSLATOR") {
    handleStopTranslator()
      .then(() => sendResponse({ ok: true }))
      .catch((error: unknown) => {
        console.error("STOP_TRANSLATOR error:", error);
        sendResponse({
          ok: false,
          error: error instanceof Error ? error.message : String(error)
        });
      });
    return true;
  }

  if (message.type === "TRANSLATOR_READY") {
    handleTranslatorReady()
      .then(() => sendResponse({ ok: true }))
      .catch((error: unknown) => {
        console.error("TRANSLATOR_READY error:", error);
        sendResponse({
          ok: false,
          error: error instanceof Error ? error.message : String(error)
        });
      });
    return true;
  }
});

async function handleStartTranslator(message: {
  targetTabId?: number;
  tabTitle?: string;
}) {
  const targetTabId = message.targetTabId;
  const tabTitle = message.tabTitle ?? "Meeting Tab";

  if (!targetTabId) {
    throw new Error("Missing targetTabId");
  }

  const translatorUrl = chrome.runtime.getURL("src/extension/translator.html");
  const existing = await chrome.tabs.query({ url: translatorUrl });

  if (existing.length > 0 && existing[0]?.id) {
    translatorTabId = existing[0].id;
    await chrome.tabs.update(translatorTabId, { active: true });
  } else {
    const tab = await chrome.tabs.create({ url: translatorUrl });
    translatorTabId = tab.id ?? null;
  }

  await chrome.storage.session.set({
    pendingTargetTabId: targetTabId,
    pendingTabTitle: tabTitle
  });
}

async function handleTranslatorReady() {
  const stored = await chrome.storage.session.get([
    "pendingTargetTabId",
    "pendingTabTitle"
  ]);

  const pendingTargetTabId = stored.pendingTargetTabId;
  const pendingTabTitle = stored.pendingTabTitle;

  if (typeof pendingTargetTabId !== "number") {
    throw new Error("No pending target tab");
  }

  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: pendingTargetTabId
  });

  if (!translatorTabId) {
    throw new Error("Translator tab not available");
  }

  await chrome.tabs.sendMessage(translatorTabId, {
    type: "START_CAPTURE_WITH_STREAM_ID",
    streamId,
    targetTabId: pendingTargetTabId,
    tabTitle:
      typeof pendingTabTitle === "string" ? pendingTabTitle : "Meeting Tab"
  });
}

async function handleStopTranslator() {
  if (!translatorTabId) return;

  try {
    await chrome.tabs.sendMessage(translatorTabId, {
      type: "STOP_CAPTURE"
    });
  } catch (error) {
    console.warn("Could not reach translator tab", error);
  }
}