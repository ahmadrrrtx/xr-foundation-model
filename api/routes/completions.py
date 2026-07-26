"""Text generation endpoints (sync + streaming SSE)."""
import json, time, logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.main import _model, _engine, _tokenizer, _model_loaded
from api.schemas import CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage
import torch

router = APIRouter()
logger = logging.getLogger("xrfm.api.completions")

@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    if not _model_loaded or _engine is None:
        raise HTTPException(503, "Model not loaded")

    # Tokenize prompt
    try:
        input_ids = torch.tensor([_tokenizer.encode(req.prompt)], dtype=torch.long)
    except Exception as e:
        raise HTTPException(400, f"Tokenization failed: {e}")

    prompt_tokens = input_ids.shape[1]

    # Generate
    try:
        output_ids = _engine.generate(
            input_ids, max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    # Decode only new tokens
    new_tokens = output_ids[prompt_tokens:]
    text = _tokenizer.decode(new_tokens.tolist())

    return CompletionResponse(
        created=int(time.time()),
        choices=[CompletionChoice(text=text, finish_reason="length")],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(new_tokens),
            total_tokens=prompt_tokens+len(new_tokens),
        ),
    )

@router.post("/v1/completions/stream")
async def completions_stream(req: CompletionRequest):
    if not _model_loaded or _engine is None:
        raise HTTPException(503, "Model not loaded")

    input_ids = torch.tensor([_tokenizer.encode(req.prompt)], dtype=torch.long)
    prompt_tokens = input_ids.shape[1]

    async def generate_stream():
        from inference.sampling import sample_token
        _model.eval()
        generated = input_ids.clone()
        past_kv = None
        next_token = None

        # Prefill: process prompt
        with torch.no_grad():
            logits, past_kv = _model(input_ids, use_cache=True)
            next_logits = logits[:, -1, :]
            if req.temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                from inference.sampling import sample_token as st
                next_token = st(next_logits, req.temperature, req.top_k, req.top_p)
        generated = torch.cat([generated, next_token], dim=1)
        token_text = _tokenizer.decode([next_token.item()])
        yield f"data: {json.dumps({'token': {'text': token_text}})}\n\n"

        for _ in range(1, req.max_new_tokens):
            with torch.no_grad():
                logits, past_kv = _model(next_token, past_key_values=past_kv, use_cache=True)
            next_logits = logits[:, -1, :]
            if req.temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                from inference.sampling import sample_token as st
                next_token = st(next_logits, req.temperature, req.top_k, req.top_p)
            generated = torch.cat([generated, next_token], dim=1)
            token_text = _tokenizer.decode([next_token.item()])
            yield f"data: {json.dumps({'token': {'text': token_text}})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")
