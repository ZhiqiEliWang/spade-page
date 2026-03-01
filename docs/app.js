import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/+esm";

const FIXED_SPACE_ID = "ZhiqiEliWang/SPADE";
const EXAMPLE_MESSAGES_GH_PAGE = [
  "Congratulations! You've been selected for a chance to get a brand new iPhone for just $1. Click here to claim now!",
  "Dear user, your account has been compromised. Please reset your password immediately by clicking on this link.",
  "You have won a free vacation to the Bahamas! Call now to claim your prize.",
  "This is a reminder that your subscription will expire soon. Please renew to continue enjoying our services.",
];

const messageInput = document.getElementById("message");
const runBtn = document.getElementById("runBtn");
const exampleMessages = document.getElementById("exampleMessages");
const scamVerdict = document.getElementById("scamVerdict");
const detectorOutput = document.getElementById("detectorOutput");
const explainerOutput = document.getElementById("explainerOutput");
const statusEl = document.getElementById("status");

let clientCache = null;
let cachedSpaceId = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.className = isError ? "status error" : "status";
}

async function getClient(spaceId) {
  if (clientCache && cachedSpaceId === spaceId) {
    return clientCache;
  }
  clientCache = await Client.connect(spaceId);
  cachedSpaceId = spaceId;
  return clientCache;
}

function normalizeText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function decodeVisibleEscapes(text) {
  return text
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t");
}

function normalizeFinalText(value) {
  const text = normalizeText(value);
  if (typeof text !== "string") return text;
  if (!text.includes("\\n") && !text.includes("\\t") && !text.includes("\\r")) return text;
  return decodeVisibleEscapes(text);
}

function setScamVerdict(state, note) {
  if (!scamVerdict) return;

  const chipTextByState = {
    scam: "Scam Detected",
    legit: "Not A Scam",
    unknown: "Scam Verdict Unknown",
    pending: "Waiting For Detector",
  };
  const chipText = chipTextByState[state] || chipTextByState.unknown;
  const detail = note || "Could not read a valid \"scam\" value from detector output.";

  scamVerdict.className = `scam-verdict ${state}`;
  scamVerdict.innerHTML = `
    <span class="scam-chip">${chipText}</span>
    <p class="scam-note">${detail}</p>
  `;
}

function coerceScamFlag(value) {
  if (value === true) return 1;
  if (value === false) return 0;
  if (typeof value === "number" && (value === 0 || value === 1)) return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "1" || normalized === "true") return 1;
    if (normalized === "0" || normalized === "false") return 0;
  }
  return null;
}

function extractDetectorObject(detector) {
  if (detector && typeof detector === "object" && !Array.isArray(detector)) {
    return detector;
  }

  const text = normalizeText(detector).trim();
  if (!text) return null;

  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch (_) {
    // Ignore parse failures on streaming partial JSON.
  }

  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try {
    const parsed = JSON.parse(match[0]);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch (_) {
    // Ignore parse failures on non-final chunks.
  }
  return null;
}

function updateScamVerdict(detector) {
  const parsed = extractDetectorObject(detector);
  if (!parsed || !Object.prototype.hasOwnProperty.call(parsed, "scam")) {
    setScamVerdict("pending", "Scam verdict appears after detector output is finalized.");
    return;
  }

  const scamFlag = coerceScamFlag(parsed.scam);
  if (scamFlag === 1) {
    setScamVerdict("scam", "Detector returned \"scam\": 1.");
    return;
  }
  if (scamFlag === 0) {
    setScamVerdict("legit", "Detector returned \"scam\": 0.");
    return;
  }
  setScamVerdict("unknown", "Could not read a valid \"scam\" value from detector output.");
}

function renderOutputs(data, isFinal = false) {
  if (!Array.isArray(data)) return;
  const [detector, explanation] = data;
  detectorOutput.textContent = normalizeText(detector);
  explainerOutput.textContent = isFinal ? normalizeFinalText(explanation) : normalizeText(explanation);
  updateScamVerdict(detector);
}

function hasMeaningfulOutput(data) {
  if (!Array.isArray(data)) return false;
  const [detector, explanation] = data;
  return normalizeText(detector).trim().length > 0 || normalizeText(explanation).trim().length > 0;
}

function queueStatusMessage(message) {
  const stage = message?.stage || message?.status?.stage || null;
  const rankRaw = message?.rank ?? message?.queue_position ?? message?.position ?? message?.status?.rank;
  const queueSizeRaw = message?.queue_size ?? message?.status?.queue_size;
  const rankNum = rankRaw == null ? NaN : Number(rankRaw);
  const queueSizeNum = queueSizeRaw == null ? NaN : Number(queueSizeRaw);
  const rank = Number.isFinite(rankNum) ? rankNum + 1 : null;
  const queueSize = Number.isFinite(queueSizeNum) ? queueSizeNum : null;

  if (rank != null && rank > 1) {
    if (queueSize != null) {
      return `Queued... (${rank}/${queueSize})`;
    }
    return `Queued... (position ${rank})`;
  }

  if (stage === "pending" || stage === "queued") {
    if (rank != null && queueSize != null) {
      return `Queued... (${rank}/${queueSize})`;
    }
    if (rank != null) {
      return `Queued... (position ${rank})`;
    }
    return "Queued...";
  }
  if (stage === "generating") return "Processing...";
  if (stage === "processing") return "Processing...";
  return null;
}

function renderExampleMessages() {
  if (!exampleMessages) return;

  exampleMessages.innerHTML = "";
  if (!Array.isArray(EXAMPLE_MESSAGES_GH_PAGE) || EXAMPLE_MESSAGES_GH_PAGE.length === 0) {
    exampleMessages.hidden = true;
    return;
  }

  const fragment = document.createDocumentFragment();
  EXAMPLE_MESSAGES_GH_PAGE.forEach((message) => {
    const text = typeof message === "string" ? message.trim() : "";
    if (!text) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "example-message-btn";
    button.textContent = text;
    button.addEventListener("click", () => {
      messageInput.value = text;
      messageInput.focus();
    });
    fragment.appendChild(button);
  });

  if (!fragment.childNodes.length) {
    exampleMessages.hidden = true;
    return;
  }

  exampleMessages.hidden = false;
  exampleMessages.appendChild(fragment);
}

async function runPipeline() {
  const text = messageInput.value.trim();

  if (!text) {
    setStatus("Please enter input text.", true);
    return;
  }

  runBtn.disabled = true;
  setStatus("Running...");
  detectorOutput.textContent = "";
  explainerOutput.textContent = "";
  setScamVerdict("pending", "Scam verdict appears after detector output is finalized.");

  try {
    const app = await getClient(FIXED_SPACE_ID);
    const job = app.submit("/pipeline", { text });
    let streamed = false;
    let lastData = null;
    const supportsStreaming = typeof job?.[Symbol.asyncIterator] === "function";

    if (supportsStreaming) {
      for await (const message of job) {
        if (message.type === "data") {
          lastData = message.data;
          renderOutputs(message.data, false);
          if (hasMeaningfulOutput(message.data)) {
            streamed = true;
            setStatus("Streaming...");
          }
        } else if (message.type === "status") {
          const statusMessage = queueStatusMessage(message);
          if (statusMessage) setStatus(statusMessage);
        }
      }
      if (lastData) {
        renderOutputs(lastData, true);
      }
    } else {
      const result = await job;
      if (result?.data) {
        renderOutputs(result.data, true);
      } else if (streamed && lastData) {
        renderOutputs(lastData, true);
      }
    }
    setStatus("Done.");
  } catch (err) {
    setStatus(`Request failed: ${err.message || String(err)}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

renderExampleMessages();
setScamVerdict("pending", "Scam verdict appears after detector output is finalized.");
runBtn.addEventListener("click", runPipeline);
