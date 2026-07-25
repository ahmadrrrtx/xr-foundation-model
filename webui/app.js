// XRFM Chat UI
const API = '/v1/completions';
const STREAM_API = '/v1/completions/stream';
let conversation = [];

function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
  document.getElementById('theme-toggle').textContent = current === 'dark' ? '🌙' : '☀️';
  localStorage.setItem('xrfm-theme', current === 'dark' ? 'light' : 'dark');
}

document.getElementById('temperature').addEventListener('input', (e) => {
  document.getElementById('temp-val').textContent = e.target.value;
});

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.textContent = text;
  document.getElementById('messages').appendChild(div);
  document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
  return div;
}

function clearChat() {
  conversation = [];
  document.getElementById('messages').innerHTML = '';
}

async function fetchModelInfo() {
  try {
    const r = await fetch('/v1/models');
    const models = await r.json();
    if (models.length > 0) {
      const m = models[0];
      document.getElementById('model-info').textContent = `Params: ${m.parameter_count.toLocaleString()} | Vocab: ${m.vocab_size} | MaxLen: ${m.max_seq_len}`;
    }
  } catch(e) {
    document.getElementById('model-info').textContent = 'API offline';
  }
}

async function sendMessage() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  addMessage('user', prompt);
  document.getElementById('prompt').value = '';
  conversation.push({role:'user', content:prompt});

  const ctx = conversation.map(m => m.content).join('\n');
  const reqBody = {
    prompt: ctx,
    max_new_tokens: parseInt(document.getElementById('max-tokens').value) || 256,
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k: parseInt(document.getElementById('top-k').value) || null,
    top_p: parseFloat(document.getElementById('top-p').value) || null,
    stream: false
  };

  try {
    const r = await fetch(API, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(reqBody) });
    const data = await r.json();
    const text = data.choices[0].text;
    addMessage('assistant', text);
    conversation.push({role:'assistant', content:text});
  } catch(e) {
    addMessage('assistant', 'Error: ' + e.message);
  }
}

async function sendStream() {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  addMessage('user', prompt);
  document.getElementById('prompt').value = '';

  const reqBody = {
    prompt: prompt,
    max_new_tokens: parseInt(document.getElementById('max-tokens').value) || 256,
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k: parseInt(document.getElementById('top-k').value) || null,
    top_p: parseFloat(document.getElementById('top-p').value) || null,
    stream: true
  };

  const msgDiv = addMessage('assistant', '');
  let fullText = '';

  try {
    const r = await fetch(STREAM_API, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(reqBody) });
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data: ') && !line.includes('[DONE]')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token && data.token.text) {
              fullText += data.token.text;
              msgDiv.textContent = fullText;
              document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
            }
          } catch(e) {}
        }
      }
    }
    conversation.push({role:'assistant', content:fullText});
  } catch(e) {
    msgDiv.textContent = 'Stream error: ' + e.message;
  }
}

// Init
fetchModelInfo();
const saved = localStorage.getItem('xrfm-theme');
if (saved) document.documentElement.setAttribute('data-theme', saved);
document.getElementById('theme-toggle').textContent = (saved || 'dark') === 'dark' ? '☀️' : '🌙';
document.getElementById('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
