// XR AI Agent — Master Web Client & Vercel Standalone Engine

let activeMode = 'chat'; // 'chat' or 'search'
let indexedDocs = [
  {
    doc_id: 'xrfm_overview',
    title: 'XRFM Architecture Overview',
    text: 'XRFM (XR Foundation Model) is built with RoPE position embeddings, SwiGLU activations, RMSNorm pre-normalization, and weight-tied embeddings. It features an integrated hybrid BM25 + Vector local search engine.'
  },
  {
    doc_id: 'xrfm_benchmarks',
    title: 'XR Foundation Model Benchmarks',
    text: 'XRFM achieves competitive perplexity scores on TinyShakespeare and Wikipedia benchmarks, running up to 1,000x faster BPE tokenization and optimized INT4 group-wise quantization.'
  },
  {
    doc_id: 'vercel_deploy_guide',
    title: 'Vercel Deployment Specs',
    text: 'XR AI Web Agent can be deployed directly to Vercel as a static single-page app or connected to a live Python FastAPI / PyTorch server running on localhost or cloud GPU instance.'
  }
];

// UI Initialization
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  updateIndexedCount();
});

function setupEventListeners() {
  const promptInput = document.getElementById('prompt-input');
  
  promptInput.addEventListener('input', () => {
    promptInput.style.height = 'auto';
    promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + 'px';
  });

  promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  document.getElementById('temperature').addEventListener('input', (e) => {
    document.getElementById('temp-val').textContent = e.target.value;
  });
  document.getElementById('max-tokens').addEventListener('input', (e) => {
    document.getElementById('tokens-val').textContent = e.target.value;
  });
  document.getElementById('top-p').addEventListener('input', (e) => {
    document.getElementById('topp-val').textContent = e.target.value;
  });
}

function setMode(mode) {
  activeMode = mode;
  document.getElementById('mode-chat').classList.toggle('active', mode === 'chat');
  document.getElementById('mode-search').classList.toggle('active', mode === 'search');
  document.getElementById('mode-badge').textContent = mode === 'chat' ? 'Chat Mode' : 'Grounded Search (RAG)';
}

function useSuggestion(text) {
  document.getElementById('prompt-input').value = text;
  if (text.toLowerCase().includes('search')) {
    setMode('search');
  }
  handleSend();
}

function clearChat() {
  const container = document.getElementById('messages-container');
  container.innerHTML = `
    <div class="welcome-card">
      <div class="hero-logo-box">
        <img src="/logo.png" alt="XR Agent Logo" class="hero-logo">
        <div class="hero-pulse"></div>
      </div>
      <h2 class="hero-title">XR AI Agent</h2>
      <p class="hero-subtitle">Chat cleared. Ready for your next request.</p>
    </div>
  `;
}

function updateIndexedCount() {
  document.getElementById('indexed-count').textContent = `${indexedDocs.length} Documents`;
}

function openSettingsModal() { document.getElementById('settings-modal').style.display = 'flex'; }
function closeSettingsModal() { document.getElementById('settings-modal').style.display = 'none'; }
function openIndexModal() { document.getElementById('index-modal').style.display = 'flex'; }
function closeIndexModal() { document.getElementById('index-modal').style.display = 'none'; }

function toggleProviderFields() {
  const provider = document.getElementById('api-provider').value;
  document.getElementById('endpoint-group').style.display = provider === 'standalone' ? 'none' : 'flex';
  document.getElementById('apikey-group').style.display = provider === 'custom_openai' ? 'flex' : 'none';
}

function saveSettings() {
  const provider = document.getElementById('api-provider').value;
  const endpoint = document.getElementById('api-endpoint').value;
  localStorage.setItem('xr_provider', provider);
  localStorage.setItem('xr_endpoint', endpoint);
  document.getElementById('backend-status').textContent = provider === 'standalone' ? 'XR Core Active' : 'Connected';
  closeSettingsModal();
}

function handleIndexDocument() {
  const docId = document.getElementById('doc-id-input').value.trim();
  const title = document.getElementById('doc-title-input').value.trim();
  const text = document.getElementById('doc-text-input').value.trim();

  if (!docId || !text) {
    alert('Document ID and Text content are required.');
    return;
  }

  indexedDocs.push({ doc_id: docId, title: title || docId, text: text });
  updateIndexedCount();
  alert(`Successfully indexed "${title || docId}" into XR Search Engine!`);
  closeIndexModal();
}

// Send Message Handler
async function handleSend() {
  const promptInput = document.getElementById('prompt-input');
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  promptInput.value = '';
  promptInput.style.height = 'auto';

  const welcomeCard = document.querySelector('.welcome-card');
  if (welcomeCard) welcomeCard.remove();

  appendMessage('user', prompt);
  const assistantBubble = appendMessage('assistant', 'Thinking...');

  const provider = localStorage.getItem('xr_provider') || 'standalone';
  const endpoint = localStorage.getItem('xr_endpoint') || 'http://localhost:8000';

  if (provider === 'xrfm_local') {
    await sendLocalXRFMRequest(prompt, endpoint, assistantBubble);
  } else {
    await sendVercelAIRequest(prompt, assistantBubble);
  }
}

function appendMessage(role, text, sources = []) {
  const container = document.getElementById('messages-container');
  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${role}`;
  
  if (role === 'assistant') {
    bubble.innerHTML = formatMarkdown(text);
    if (sources && sources.length > 0) {
      const srcBox = document.createElement('div');
      srcBox.className = 'source-box';
      srcBox.innerHTML = `<strong>Sources Cited:</strong> ` + sources.map(s => `<em>[${s.title}]</em>`).join(', ');
      bubble.appendChild(srcBox);
    }
  } else {
    bubble.textContent = text;
  }

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

// Vercel Serverless AI Call
async function sendVercelAIRequest(prompt, bubbleElement) {
  try {
    const apiPath = activeMode === 'search' ? '/api/search' : '/api/generate';
    const body = activeMode === 'search' 
      ? { query: prompt, documents: indexedDocs } 
      : { prompt, temperature: parseFloat(document.getElementById('temperature').value) };

    const res = await fetch(apiPath, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const answer = data.answer || 'No response generated.';
    const sources = data.sources || [];

    await typeText(bubbleElement, answer, sources);
  } catch (err) {
    console.warn('Vercel API fallback:', err);
    // Fallback to local response engine
    await sendFallbackAIResponse(prompt, bubbleElement);
  }
}

async function sendLocalXRFMRequest(prompt, endpoint, bubbleElement) {
  try {
    const apiPath = activeMode === 'search' ? '/v1/search/query' : '/v1/completions';
    const body = activeMode === 'search' ? { query: prompt } : { prompt: prompt, max_new_tokens: 256 };

    const res = await fetch(endpoint + apiPath, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    const data = await res.json();
    const answer = activeMode === 'search' ? data.answer : data.choices[0].text;
    const sources = activeMode === 'search' ? data.sources : [];

    bubbleElement.innerHTML = formatMarkdown(answer);
    if (sources && sources.length > 0) {
      const srcBox = document.createElement('div');
      srcBox.className = 'source-box';
      srcBox.innerHTML = `<strong>Sources Cited:</strong> ` + sources.map(s => `<em>[${s.title}]</em>`).join(', ');
      bubbleElement.appendChild(srcBox);
    }
  } catch (err) {
    bubbleElement.innerHTML = `<span style="color:#FF4949;">Local Server Offline: ${err.message}. Defaulting to Vercel XR Core.</span>`;
    await sendVercelAIRequest(prompt, bubbleElement);
  }
}

async function sendFallbackAIResponse(prompt, bubbleElement) {
  const query = prompt.toLowerCase();
  let answer = "";

  if (query.includes('what is xr') || query.includes('xrfm') || query.includes('who created')) {
    answer = "I am **XR**, the AI Agent You Can Actually Trust.\n\nI am powered by the **XR Foundation Model (XRFM)** architecture, built with Rotary Position Embeddings (RoPE), SwiGLU activations, RMSNorm pre-normalization, and an integrated Grounded RAG Search Engine.";
  } else if (query.includes('architecture') || query.includes('rope') || query.includes('swiglu')) {
    answer = "**XRFM Architecture Specifications:**\n\n1. **RoPE (Rotary Position Embedding):** Encodes relative positions directly into key-query attention.\n2. **SwiGLU Activation:** Gated feed-forward layer providing high representational capacity.\n3. **RMSNorm:** Fast root-mean-square layer normalization.\n4. **Weight Tying:** Shared parameters between embedding matrix and LM head.";
  } else if (query.includes('deploy') || query.includes('vercel')) {
    answer = "**Connecting Real AI Models on Vercel:**\n\n1. Get a **Free Groq API Key** from [console.groq.com](https://console.groq.com).\n2. In Vercel Project Settings -> Environment Variables, add `GROQ_API_KEY` = your_key.\n3. Redeploy — XR will answer all complex questions live via Llama-3.1 70B!";
  } else {
    answer = `I have received your request: "${prompt}".\n\nTo enable full live LLM inference on Vercel, add a free \`GROQ_API_KEY\` or \`OPENAI_API_KEY\` in your Vercel Project Settings -> Environment Variables!`;
  }

  await typeText(bubbleElement, answer);
}

async function typeText(element, text, sources = []) {
  element.innerHTML = '';
  const chars = text.split('');
  let current = '';
  
  for (let i = 0; i < chars.length; i += 3) {
    current += chars.slice(i, i + 3).join('');
    element.innerHTML = formatMarkdown(current);
    document.getElementById('messages-container').scrollTop = document.getElementById('messages-container').scrollHeight;
    await new Promise(r => setTimeout(r, 10));
  }

  if (sources && sources.length > 0) {
    const srcBox = document.createElement('div');
    srcBox.className = 'source-box';
    srcBox.innerHTML = `<strong>Sources Cited:</strong> ` + sources.map(s => `<em>[${s.title}]</em>`).join(', ');
    element.appendChild(srcBox);
  }
}

function formatMarkdown(text) {
  return text
    .replace(/```python([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/```bash([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1$</strong>')
    .replace(/\n/g, '<br>');
}
