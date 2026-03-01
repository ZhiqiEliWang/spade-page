import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/+esm";

const FIXED_SPACE_ID = "ZhiqiEliWang/SPADE";

const messageInput = document.getElementById("message");
const runBtn = document.getElementById("runBtn");
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

function renderOutputs(data, isFinal = false) {
  if (!Array.isArray(data)) return;
  const [detector, explanation] = data;
  detectorOutput.textContent = normalizeText(detector);
  explainerOutput.textContent = isFinal ? normalizeFinalText(explanation) : normalizeText(explanation);
}

function queueStatusMessage(message) {
  const stage = message?.stage || message?.status?.stage || null;
  const rankRaw = message?.rank ?? message?.queue_position ?? message?.position ?? message?.status?.rank;
  const queueSizeRaw = message?.queue_size ?? message?.status?.queue_size;
  const rankNum = rankRaw == null ? NaN : Number(rankRaw);
  const queueSizeNum = queueSizeRaw == null ? NaN : Number(queueSizeRaw);
  const rank = Number.isFinite(rankNum) ? rankNum + 1 : null;
  const queueSize = Number.isFinite(queueSizeNum) ? queueSizeNum : null;

  if (stage === "pending" || stage === "queued") {
    if (rank != null && queueSize != null) {
      return `Queued... (${rank}/${queueSize})`;
    }
    if (rank != null) {
      return `Queued... (position ${rank})`;
    }
    return "Queued...";
  }
  if (stage === "generating") return "Streaming...";
  if (stage === "processing") return "Processing...";
  return null;
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
          streamed = true;
          setStatus("Streaming...");
        } else if (message.type === "status") {
          const statusMessage = queueStatusMessage(message);
          if (statusMessage) setStatus(statusMessage);
        }
      }
    }

    const result = await job;
    if (result?.data) {
      renderOutputs(result.data, true);
    } else if (streamed && lastData) {
      renderOutputs(lastData, true);
    }
    setStatus("Done.");
  } catch (err) {
    setStatus(`Request failed: ${err.message || String(err)}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runPipeline);
