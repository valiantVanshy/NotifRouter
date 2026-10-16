# notif-router

**An AI agent that decides which WhatsApp messages deserve your attention.**

Every incoming message (text, image poster or screenshot, or voice note) is routed to one of three outcomes:

| Action   | Meaning                                                   |
|----------|-----------------------------------------------------------|
| `notify` | Interrupt the user now                                    |
| `digest` | Useful but not urgent, so batch it for later                    |
| `mute`   | Low value, repetitive, unwanted, suspicious or unsafe     |

Decisions are **personalised**. The same sale poster can be useful to one user and noise to another, a payment reminder is fine from a trusted group admin but risky from a new sender, and a muted family group can still break through with a direct @mention.

---

## How it works

```
                 ┌──────────────────────────┐
 message ──────► │ 1. Safety guardrails     │── scam / spoof / injection ──► mute (no LLM cost)
                 └────────────┬─────────────┘
                              │ (may lock action = mute, e.g. muted group)
                 ┌────────────▼─────────────┐
                 │ 2. Decision cache (DB)   │── same input + model + prompt ──► reuse
                 └────────────┬─────────────┘
                 ┌────────────▼─────────────┐   tools
                 │ 3. Tool-calling agent    │◄──────────► get_sender_context
                 │ (OpenAI-compatible LLM)  │             get_recipient_profile
                 │                          │             search_message_history
                 │                          │             analyze_media (Whisper / vision)
                 └────────────┬─────────────┘             submit_decision (validated)
                              │ on failure / no API key
                 ┌────────────▼─────────────┐
                 │ 4. Heuristic fallback    │
                 └────────────┬─────────────┘
                              ▼
            decision + reason + confidence + evidence  ──►  SQLite (with tool trace)
```

1. **Guardrails** (`guardrails.py`): deterministic rules for things an LLM should never be trusted to get wrong:
   - prompt-injection attempts
   - OTP, password or card harvesting
   - unverified businesses spoofing a brand's domain
   - new, heavily reported business accounts
   - promotions to users who opted out
   - muted groups
   - repeatedly ignored forwards
2. **Agent** (`agent/`): a hand-rolled tool-calling loop, with no framework. The model investigates who sent the message, how this user reacted to similar past messages, and what the media says, then must call `submit_decision`. Each submission is validated:
   - Values must be allowed ones.
   - Evidence IDs must come from history the agent actually retrieved.
   - Guardrail-locked actions can't be overridden.

   A bad submission goes back to the model as a tool error so it can correct itself.
3. **History search** (`context/history_search.py`) ranks the user's past messages using:
   - text similarity
   - whether they came through the same chat, business or sender
   - how the user reacted (replied, opened, dismissed, reported, muted)

   It never uses messages from after the incoming one.
4. **Heuristic fallback** (`heuristics.py`) keeps the service answering during an OpenAI outage, and powers a fully offline mode.
5. **Untrusted content handling**: message text, transcripts and image text are treated as data, never as instructions.

## Project layout

```
code/
├── main.py                     # `python main.py` → routes dataset/messages.csv
├── notif_router/
│   ├── config.py               # settings from env / .env
│   ├── schemas.py              # Action, MessageType, IncomingMessage, RoutingDecision
│   ├── context/
│   │   ├── store.py            # loads and indexes users, groups, businesses, history
│   │   └── history_search.py   # evidence retrieval and ranking
│   ├── guardrails.py           # deterministic safety rules
│   ├── heuristics.py           # offline / fallback router
│   ├── media.py                # voice transcription and image OCR, cached in the DB
│   ├── agent/
│   │   ├── llm.py              # OpenAI client behind a small protocol (mockable)
│   │   ├── prompts.py          # system prompt and PROMPT_VERSION
│   │   ├── tools.py            # tool schemas and implementations
│   │   └── router_agent.py     # the agent loop and submission validation
│   ├── router.py               # orchestration: guardrails → cache → agent → fallback → DB
│   ├── storage/                # SQLAlchemy tables and repository
│   ├── api/app.py              # FastAPI service
│   ├── evaluation.py           # batch routing and accuracy report
│   └── cli.py                  # `notif-router route | evaluate | serve`
└── tests/                      # pytest suite, runs offline with a scripted fake LLM
```

## Setup

Requires Python 3.10+.

```bash
cd code
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then put your API key in .env
```

The router works with any **OpenAI-compatible** provider. `.env.example` is preset for
**Groq's free tier**: Qwen 3.8 27B for the agent and images (with `gpt-oss-120b` and `gpt-oss-20b` as fallback models) and Whisper for voice
notes. Switching to OpenAI is a matter of changing the key and removing the base URL.

Without an API key everything still runs in **offline mode** (guardrails + heuristics).

## Usage

### Route the dataset (writes `dataset/output.csv`)

```bash
python main.py                    # agent mode if an API key is set
python main.py --offline          # no LLM calls
python -m notif_router.cli route --input my.csv --output predictions.csv
```

Re-runs are cheap. Agent decisions and media transcripts are cached in SQLite, keyed by the exact input, model and prompt version. Use `--no-cache` to force fresh calls.

### Measure accuracy on the labelled samples

```bash
python -m notif_router.cli evaluate            # agent
python -m notif_router.cli evaluate --offline  # heuristics only
```

Results on `dataset/sample_messages.csv` (30 labelled messages):

| Mode | Action accuracy | Type accuracy | Evidence overlap |
|---|---|---|---|
| Offline (guardrails + heuristics) | 96.7% | 76.7% | 48.4% |
| Agent, prompt v1 (Groq `gpt-oss-120b`) | 93.3% | 76.7% | 58.1% |
| **Agent, prompt v2** | **100%** | **90.0%** | **58.1%** |
| Agent, prompt v3 with context pre-fetch (`gpt-oss-20b`) | 96.7% | 80.0% | 61.3% |
| **Agent, prompt v3 (`qwen/qwen3.8-27b`), the default** | **100%** | **93.3%** | **67.7%** |

What each iteration fixed:
- **v1 → v2:** reading the agent's tool traces on its mistakes found a misleading tool field and vague type boundaries. It also found a provider quirk: Groq rejects valid decisions written outside the tool call, and the client now recovers those. With only 30 labelled examples, the prompt changes were kept to general definitions so they don't overfit the samples.
- **v2 → v3 (cost):** v2 spent 3–4 tool round trips per message gathering the same context, and each round trip resent the whole conversation. v3 gathers sender, recipient, media and history in code and puts them in the first message, so the agent usually decides in **about one LLM call and 2.8–3.6k tokens** (measured). The tools remain available for follow-up lookups.
- **Model choice:** the same v3 prompt was compared across the models available on the free tier. Qwen scored best and became the default, with the gpt-oss models as fallbacks when its daily quota runs out.

### Final run on `dataset/messages.csv`

All 110 messages were routed with no offline fallbacks:
- **By whom:** 81 by the agent (13 Qwen, 4 `gpt-oss-120b`, 64 `gpt-oss-20b` as the daily quotas of the stronger models ran out) and 29 by the safety guardrails.
- **Outcome:** 55 mute, 32 notify, 23 digest.
- **Automated checks:** the output has exactly one row per message, valid values, and evidence IDs that exist in history. Every scam or spam is muted, and no message containing credential or phishing language is set to notify.
- **Consistency check:** a post-decision check moves notify to digest when the sender explicitly says the message is not urgent. It changed 1 decision.

### Run the API

```bash
python -m notif_router.cli serve --host 0.0.0.0 --port 8000
# interactive docs at http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/v1/route \
  -H "Content-Type: application/json" -H "X-API-Key: $NOTIF_ROUTER_API_KEY" \
  -d '{"message_id":"m1","user_id":"u_009","conversation_type":"personal",
       "sender_user_id":"u_050","created_at":"2026-07-20 10:00",
       "message_text":"Your wallet KYC failed. Reply with the OTP to keep payments active."}'
```

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Liveness and mode (`agent` / `offline`) |
| POST | `/v1/route` | Route one message |
| POST | `/v1/route/batch` | Route up to 100 messages |
| GET  | `/v1/decisions?user_id=&limit=` | Recent decisions |
| GET  | `/v1/decisions/{message_id}` | Latest decision for a message |
| POST | `/v1/decisions/{decision_id}/feedback` | Store a user correction |
| GET  | `/v1/stats` | Counts by action, feedback volume |

When `NOTIF_ROUTER_API_KEY` is set, every endpoint except `/health` requires the `X-API-Key` header.

### Tests and lint

```bash
pytest
ruff check .
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NOTIF_ROUTER_LLM_API_KEY` (or `OPENAI_API_KEY`) | none | Enables the agent; offline mode without it |
| `NOTIF_ROUTER_LLM_BASE_URL` | OpenAI | Any OpenAI-compatible API, e.g. `https://api.groq.com/openai/v1` |
| `NOTIF_ROUTER_MODEL` | `gpt-4o` | Agent model (must support tool calling) |
| `NOTIF_ROUTER_FALLBACK_MODELS` | none | Comma-separated models used when the main one hits a daily quota |
| `NOTIF_ROUTER_VISION_MODEL` | `gpt-4o` | Image description and OCR |
| `NOTIF_ROUTER_TRANSCRIPTION_MODEL` | `whisper-1` | Voice notes |
| `NOTIF_ROUTER_AGENT_MAX_STEPS` | `6` | Tool-call rounds before the agent must decide |
| `NOTIF_ROUTER_LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `NOTIF_ROUTER_LLM_RATE_LIMIT_MAX_WAIT_SECONDS` | `20` | How long a call may wait out per-minute rate limits (batch commands use 600) |
| `NOTIF_ROUTER_DATASET_DIR` | `../dataset` | Context data location |
| `NOTIF_ROUTER_DATABASE_URL` | `sqlite:///code/notif_router.db` | Any SQLAlchemy URL (e.g. Postgres) |
| `NOTIF_ROUTER_API_KEY` | none | Require `X-API-Key` on API calls |
| `NOTIF_ROUTER_LOG_LEVEL` | `INFO` | Logs are emitted as JSON lines |

## Design decisions

- **Guardrails before the LLM.** Scam and spoofing checks are cheap, deterministic and immune to prompt injection, and they save LLM calls.
- **Tools bound to the current message.** The model never passes user or business IDs, so it can't look up the wrong person or invent one.
- **Evidence must be retrieved, not invented.** Cited message IDs are checked against what the search tool returned in that run.
- **Rate limits are handled by kind.**
  - Per-minute limits: the client waits as long as the provider asks.
  - A spent daily quota: the client switches to a fallback model, and each decision records which model made it.
  - Out of credit, or an invalid key: the router switches to offline mode instead of failing every request.
- **Never fail a request.** Any agent or API error falls back to the heuristic router, and the `source` field records which path decided.
- **Reproducible.** Temperature 0, a fixed seed, versioned prompts and an input-hash cache.
- **Feedback loop.** User corrections are stored next to the decision and its tool trace, ready for evaluation or prompt tuning.
