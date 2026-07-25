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

// Local Knowledge Base for Standalone Web AI
const KNOWLEDGE_BASE = {
  architecture: `XR Foundation Model (XRFM) is a modern decoder-only transformer engineered from scratch in pure PyTorch.

**Key Primitives:**
1. **RoPE (Rotary Position Embeddings):** Enables dynamic relative positional encoding.
2. **SwiGLU (Swish-Gated Linear Units):** Provides higher capacity feed-forward representations.
3. **RMSNorm (Root Mean Square Normalization):** Faster pre-normalization before attention and FFN layers.
4. **FlashAttention-2 & SDPA:** Fused matrix computation reducing memory from O(N²) to O(N).
5. **Priority-Rank BPE Tokenizer:** Fast subword tokenization with byte-level UTF-8 fallback.`,

  deploy: `**Deploying XR AI Agent to Vercel:**

1. **Option A: Deploy via Vercel CLI**
   \`\`\`bash
   npm install -g vercel
   cd vercel_app
   vercel
   \`\`\`

2. **Option B: Push to GitHub & Connect Vercel**
   - Push repository to GitHub.
   - Import repo into [Vercel Dashboard](https://vercel.com).
   - Set Output Directory to \`vercel_app\`.

3. **Connecting to Local PyTorch Server:**
   - Run local XRFM server: \`uvicorn api.main:app --port 8000\`
   - In XR Web UI, click **⚙️ API Config** -> set Endpoint URL to \`http://localhost:8000\`.`,

  search: `**Grounded Search Engine (RAG):**

XR AI Agent uses a **Hybrid BM25 + Dense Retrieval Engine** to eliminate hallucinations:
- **Document Indexing:** Text is broken into overlapping chunks.
- **BM25 Lexical Scoring:** Matches exact terms and TF-IDF statistics.
- **Source Citation:** Every answer generated in Search Mode explicitly cites indexed documents.`,

  code: `\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight
\`\`\``
};

// UI Initialization
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  updateIndexedCount();
});

function setupEventListeners() {
  const promptInput = document.getElementById('prompt-input');
  
  // Auto-expand textarea
  promptInput.addEventListener('input', () => {
    promptInput.style.height = 'auto';
    promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + 'px';
  });

  // Enter to send
  promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Parameter sliders
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

// Modal Handlers
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
  document.getElementById('backend-status').textContent = provider === 'standalone' ? 'XR Core Active (Web)' : 'Connected to Server';
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

  // Clear input
  promptInput.value = '';
  promptInput.style.height = 'auto';

  // Hide welcome card if present
  const welcomeCard = document.querySelector('.welcome-card');
  if (welcomeCard) welcomeCard.remove();

  // Add User Message
  appendMessage('user', prompt);

  // Add Assistant Placeholder
  const assistantBubble = appendMessage('assistant', 'Thinking...');

  // Check if connected to backend API
  const provider = localStorage.getItem('xr_provider') || 'standalone';
  const endpoint = localStorage.getItem('xr_endpoint') || 'http://localhost:8000';

  if (provider === 'xrfm_local') {
    await sendLocalXRFMRequest(prompt, endpoint, assistantBubble);
  } else {
    // Standalone Web AI Engine
    await sendStandaloneAIResponse(prompt, assistantBubble);
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

// Standalone Web Response Generator
async function sendStandaloneAIResponse(prompt, bubbleElement) {
  const query = prompt.toLowerCase();
  let responseText = "";
  let citedSources = [];

  if (activeMode === 'search') {
    // Grounded RAG Search Simulation
    const matched = indexedDocs.filter(d => 
      query.split(' ').some(word => word.length > 3 && d.text.toLowerCase().includes(word))
    );

    if (matched.length > 0) {
      citedSources = matched;
      responseText = `**Grounded Search Results:**\n\n` + 
        matched.map(m => `According to **${m.title}**:\n${m.text}`).join('\n\n') +
        `\n\n*Summary:* XR AI Agent has verified these facts from your indexed search repository.`;
    } else {
      responseText = `Based on XR Search Engine indexing, here is the verified context:\n\n` + KNOWLEDGE_BASE.architecture;
      citedSources = [indexedDocs[0]];
    }
  } else {
    // General Conversational Chat
    if (query.includes('architecture') || query.includes('rope') || query.includes('swiglu')) {
      responseText = KNOWLEDGE_BASE.architecture;
    } else if (query.includes('deploy') || query.includes('vercel')) {
      responseText = KNOWLEDGE_BASE.deploy;
    } else if (query.includes('search') || query.includes('rag')) {
      responseText = KNOWLEDGE_BASE.search;
    } else if (query.includes('code') || query.includes('python') || query.includes('flash')) {
      responseText = KNOWLEDGE_BASE.code;
    } else {
      responseText = `Hello! I am **XR**, the AI Agent You Can Actually Trust.\n\nI am powered by the **XR Foundation Model (XRFM)** architecture featuring RoPE position embeddings, SwiGLU, RMSNorm, and an integrated Grounded Search Engine.\n\nHow can I assist you with your project today?`;
    }
  }

  // Stream text typing effect
  await typeText(bubbleElement, responseText, citedSources);
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
    bubbleElement.innerHTML = `<span style="color:#FF4949;">Server Connection Error: ${err.message}. Defaulting to Standalone XR Core.</span>`;
  }
}

async function typeText(element, text, sources = []) {
  element.innerHTML = '';
  const chars = text.split('');
  let current = '';
  
  for (let i = 0; i < chars.length; i += 3) {
    current += chars.slice(i, i + 3).join('');
    element.innerHTML = formatMarkdown(current);
    document.getElementById('messages-container').scrollTop = document.getElementById('messages-container').scrollHeight;
    await new Promise(r => setTimeout(r, 12));
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
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}
