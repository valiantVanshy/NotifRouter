# notif-router

An AI agent that decides whether each WhatsApp message should **notify** you now, wait for a **digest**, or be **muted**. It works across text, image posters and voice notes, and personalises each decision to the recipient.

- `code/`: the application (agent, API, CLI, tests). Start with [code/README.md](code/README.md).
- `dataset/`: users, chats, businesses, message history and media used as context.
- `problem_statement.md`: the original challenge brief.
- `legacy/`: the first prototype, kept for comparison (including its original `output.csv`).

Submission files:
- `code.zip`: the `code/` folder, without `.env`, the virtualenv or the local database.
- `dataset/output.csv`: predictions for all 110 messages.
- `chat_transcript.md`: the development conversation, with keys redacted.

Quick start:

```bash
cd code
python -m venv .venv && .venv/Scripts/activate   # or: source .venv/bin/activate
pip install -r requirements-dev.txt
python main.py --offline       
```
