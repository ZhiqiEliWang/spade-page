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

function renderOutputs(data) {
  if (!Array.isArray(data)) return;
  const [detector, explanation] = data;
  detectorOutput.textContent = normalizeText(detector);
  explainerOutput.textContent = normalizeText(explanation);
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
    const supportsStreaming = typeof job?.[Symbol.asyncIterator] === "function";

    if (supportsStreaming) {
      for await (const message of job) {
        if (message.type === "data") {
          renderOutputs(message.data);
          streamed = true;
          setStatus("Streaming...");
        }
      }
    }

    const result = await job;
    if (!streamed && result?.data) {
      renderOutputs(result.data);
    }
    setStatus("Done.");
  } catch (err) {
    setStatus(`Request failed: ${err.message || String(err)}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runPipeline);
