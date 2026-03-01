import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/+esm";

const spaceInput = document.getElementById("spaceId");
const messageInput = document.getElementById("message");
const runBtn = document.getElementById("runBtn");
const detectorOutput = document.getElementById("detectorOutput");
const explainerOutput = document.getElementById("explainerOutput");
const statusEl = document.getElementById("status");

const initialSpaceId = window.SPADE_CONFIG?.spaceId || "";
spaceInput.value = initialSpaceId;

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

async function runPipeline() {
  const spaceId = spaceInput.value.trim();
  const text = messageInput.value.trim();

  if (!spaceId) {
    setStatus("Please set your Hugging Face Space ID.", true);
    return;
  }
  if (!text) {
    setStatus("Please enter input text.", true);
    return;
  }

  runBtn.disabled = true;
  setStatus("Running...");

  try {
    const app = await getClient(spaceId);
    const result = await app.predict("/pipeline", { text });
    const [detector, explanation] = result.data;

    detectorOutput.textContent = JSON.stringify(detector, null, 2);
    explainerOutput.textContent = explanation;
    setStatus("Done.");
  } catch (err) {
    setStatus(`Request failed: ${err.message || String(err)}`, true);
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runPipeline);
