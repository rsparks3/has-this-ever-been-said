# Has this ever been said?

Type any text and get an estimate of how likely it is that *anyone, ever* has already said or
written exactly those words.

**Try it: https://rsparks3.github.io/has-this-ever-been-said/** (the language model runs inside
your browser; nothing you type is sent anywhere).

## How it works

1. A small open language model scores the text token by token:
   `P(text) = Π P(token_i | earlier tokens)`. No end-of-text token is scored, so this is the
   probability of the text appearing as a stretch of language, not of someone stopping exactly there.
2. The dashboard multiplies by how many words humanity has produced in a scope
   (spoken ≈ 1.4×10^19, written ≈ 10^17, public web ≈ 10^15 by default; all editable on the page)
   to get the expected number of prior occurrences λ.
3. `P(said before) = 1 − e^(−λ)`.

It also shows the running probability for every prefix of the text (where the sentence "became
unique"), shades each token by how surprising the model found it, and links to exact-phrase web
searches so the "public web" estimate can be checked.

The output is an order-of-magnitude estimate under strong assumptions (independence, a web-trained
model standing in for all human speech, exact-string matching). It is meant to be fun and
instructive, not a measurement.

## Two ways to run it

The same page (`docs/index.html`) works in two modes and picks one automatically.

**Static site (GitHub Pages, or any file host).** With no server behind it, the page loads
[Transformers.js](https://github.com/huggingface/transformers.js) and runs an ONNX copy of the model
in WebAssembly. Default model: `HuggingFaceTB/SmolLM-360M` in 4-bit (about 390 MB, downloaded from
Hugging Face once and cached by the browser). A 135M option (about 180 MB) is offered for slow
connections but is noticeably rougher.

To publish: push this repo to GitHub, then Settings → Pages → "Deploy from a branch" → branch
`main`, folder `/docs`.

**Local server (optional, more accurate).** A FastAPI server holds the full-precision
`HuggingFaceTB/SmolLM2-360M` model and the page calls it instead:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\uvicorn server:app --host 127.0.0.1 --port 8010
```

Open http://localhost:8010. Environment overrides: `EVERSAID_MODEL` (any Hugging Face causal LM
id), `EVERSAID_MAX_TOKENS` (default 512).

Add `?q=your+text` to either URL to link straight to a result.

## Notes on the in-browser model

The browser scorer reproduces `scorer.py` exactly (same start-of-text marker, same rows of logits);
at full precision the two agree to the last digit. Quantization is what matters:

| Weights | Download | log10 P on a test sentence (fp32 reference −18.6) |
|---|---|---|
| SmolLM-135M q8 (dynamic 8-bit) | 136 MB | −23.4 |
| SmolLM-135M q4 (4-bit weights, fp32 activations) | 181 MB | −22.0 (almost all of the error on the first token) |
| SmolLM-135M fp32 | 540 MB | −18.6 |
| SmolLM-360M q8 | 363 MB | −23.9 |
| SmolLM-360M q4 | 387 MB | −17.7 (fp32 reference for this model: −17.5) |

Dynamic 8-bit quantization wrecks the first position after the start marker (the "attention sink"
token has huge activations). The 360M model in q4 stays within about half an order of magnitude of
full precision on every test sentence, so it is the default. fp16 does not run on the WebAssembly
backend. Hidden testing hooks: `?model=org/name`, `?dtype=q8|q4|fp32`, `?ctx=%0A`.

## Layout

```
docs/index.html    the dashboard (works as a static page or against the server)
server.py          FastAPI: GET /  GET /api/info  POST /api/score
scorer.py          loads the model on a background thread and scores text
requirements.txt   torch (CPU wheel), transformers, fastapi, uvicorn
```
