# notif-router

**An AI agent that decides which WhatsApp messages deserve your attention.**

Every incoming message (text, image poster or voice note) is routed to one of three outcomes:

| Action   | Meaning                                                      |
|----------|--------------------------------------------------------------|
| `notify` | Interrupt the user now                                       |
| `digest` | Useful but not urgent, so batch it for later                 |
| `mute`   | Low value, repetitive, unwanted, suspicious or unsafe        |

Decisions are personalised. The same sale poster can be useful to one user and noise to another, and a muted family group can still break through with a direct @mention.

## Results

Measured on the 30 labelled samples in `dataset/sample_messages.csv`:

| Mode                                         | Action accuracy | Type accuracy |
|----------------------------------------------|-----------------|---------------|
| Offline (guardrails + heuristics, no LLM)    | 96.7%           | 76.7%         |
| Agent (Qwen 27B on Groq's free tier)         | 100%            | 93.3%         |

All 110 messages in `dataset/messages.csv` were routed with no offline fallbacks: 55 mute, 32 notify, 23 digest. Safety guardrails decided 29 of them without any LLM call, and every scam or spam message was muted.

With only 30 labelled examples, treat these as a sanity check, not a benchmark. The full iteration history is in [code/README.md](code/README.md).

## How it works

1. **Guardrails** catch scams, spoofed brands, prompt injection and explicit user preferences with deterministic rules. They are cheap and can't be talked out of a decision by message text.
2. **A tool-calling agent** looks up the sender, the recipient's profile, similar past messages and any voice or image content, then submits a validated decision. It can only cite evidence it actually retrieved.
3. **A heuristic fallback** keeps the service answering when the LLM is down, out of quota or has no API key.

Each decision comes with a reason, a confidence score and the IDs of past messages that support it.

## What's in this repo

- `code/`: the application (agent, API, CLI, tests). Start with [code/README.md](code/README.md).
- `dataset/`: users, chats, businesses, message history and media used as context.
- `problem_statement.md`: the original challenge brief.
- `legacy/`: the first prototype, kept for comparison, including its original `output.csv`.

### Submission files

- `code.zip`: the `code/` folder, without `.env`, the virtualenv or the local database.
- `dataset/output.csv`: predictions for all 110 messages.
- `chat_transcript.md`: the development conversation, with keys redacted.

## Quick start

Requires Python 3.10+.

```bash
cd code
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Try it with no API key (guardrails + heuristics only). This writes to a separate file so it doesn't overwrite the submitted predictions:

```bash
python main.py --offline --output ../predictions_offline.csv
```

To use the agent, copy `.env.example` to `.env`, add an OpenAI-compatible API key (the example is preset for Groq's free tier), then run:

```bash
python main.py
python -m notif_router.cli evaluate     # accuracy on the labelled samples
pytest                                  # tests run offline with a scripted fake LLM
```
