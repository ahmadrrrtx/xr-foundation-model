// Vercel Serverless Function — XR AI Generation Endpoint
// Automatically called by Vercel web app at /api/generate

module.exports = async (req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { prompt, temperature = 0.7, max_tokens = 512 } = req.body || {};

  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'Prompt string is required' });
  }

  // 1. Check for Groq / OpenAI API Key in Vercel Environment Variables
  const apiKey = process.env.GROQ_API_KEY || process.env.OPENAI_API_KEY || process.env.AI_API_KEY;
  const customBackend = process.env.XRFM_BACKEND_URL;

  // 2. If Custom PyTorch Backend is set in Vercel Env
  if (customBackend) {
    try {
      const response = await fetch(`${customBackend}/v1/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_new_tokens: max_tokens, temperature }),
      });
      const data = await response.json();
      return res.status(200).json({ answer: data.choices[0].text });
    } catch (err) {
      console.error('Custom backend error:', err);
    }
  }

  // 3. If GROQ API Key is configured (Free Llama-3 70B fast inference)
  if (process.env.GROQ_API_KEY) {
    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'llama-3.1-70b-versatile',
          messages: [
            { role: 'system', content: 'You are XR, a helpful, highly intelligent AI agent created by XR Foundation Model (XRFM). You are the AI Agent You Can Actually Trust.' },
            { role: 'user', content: prompt }
          ],
          temperature: temperature,
          max_tokens: max_tokens,
        }),
      });
      const data = await response.json();
      const answer = data.choices?.[0]?.message?.content || 'XR Engine temporarily unavailable.';
      return res.status(200).json({ answer });
    } catch (err) {
      console.error('Groq API error:', err);
    }
  }

  // 4. If OpenAI API Key is configured
  if (process.env.OPENAI_API_KEY) {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are XR, a helpful, intelligent AI agent created by XR Foundation Model (XRFM).' },
            { role: 'user', content: prompt }
          ],
          temperature: temperature,
          max_tokens: max_tokens,
        }),
      });
      const data = await response.json();
      const answer = data.choices?.[0]?.message?.content || 'XR Engine response error.';
      return res.status(200).json({ answer });
    } catch (err) {
      console.error('OpenAI API error:', err);
    }
  }

  // 5. Intelligent Built-in Fallback Assistant
  const query = prompt.toLowerCase();
  let answer = "";

  if (query.includes('what is xr') || query.includes('xrfm') || query.includes('who created')) {
    answer = "I am **XR**, the AI Agent You Can Actually Trust.\n\nI am powered by the **XR Foundation Model (XRFM)** architecture, built with Rotary Position Embeddings (RoPE), SwiGLU activations, RMSNorm pre-normalization, and an integrated Grounded RAG Search Engine.";
  } else if (query.includes('architecture') || query.includes('rope') || query.includes('swiglu')) {
    answer = "**XRFM Architecture Specifications:**\n\n1. **RoPE (Rotary Position Embedding):** Encodes relative positions directly into key-query attention.\n2. **SwiGLU Activation:** Gated feed-forward layer providing high representational capacity.\n3. **RMSNorm:** Fast root-mean-square layer normalization.\n4. **Weight Tying:** Shared parameters between embedding matrix and LM head.";
  } else if (query.includes('deploy') || query.includes('vercel')) {
    answer = "**Vercel AI Integration Setup:**\n\nTo connect real LLM models to this Vercel site:\n1. Get a **Free Groq API Key** from [console.groq.com](https://console.groq.com).\n2. In Vercel Project Settings -> Environment Variables, add `GROQ_API_KEY` = your_key.\n3. Redeploy on Vercel — XR will answer all complex questions live via Llama-3.1 70B!";
  } else {
    answer = `I have received your prompt: "${prompt}".\n\nTo enable full live LLM inference on Vercel:\n- Add a free \`GROQ_API_KEY\` or \`OPENAI_API_KEY\` in your Vercel Project Environment Variables, or connect to your local PyTorch server in **⚙️ API Config**!`;
  }

  return res.status(200).json({ answer });
};
