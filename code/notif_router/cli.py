"""Command-line interface.

  notif-router route     Route every message in a CSV and write predictions
  notif-router evaluate  Score the router against labelled sample messages
  notif-router serve     Run the REST API
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from notif_router.config import get_settings
from notif_router.logging_config import configure_logging

BATCH_RATE_LIMIT_WAIT_SECONDS = 600.0


def _router(offline: bool):
    from notif_router.router import NotificationRouter

    # Batch jobs have no user waiting on them, so let calls sit out rate limits for longer.
    settings = get_settings()
    wait = max(settings.llm_rate_limit_max_wait_seconds, BATCH_RATE_LIMIT_WAIT_SECONDS)
    settings = settings.model_copy(update={"llm_rate_limit_max_wait_seconds": wait})
    return NotificationRouter.from_settings(settings, offline=offline)


def cmd_route(args: argparse.Namespace) -> int:
    from notif_router.evaluation import route_file

    settings = get_settings()
    input_path = Path(args.input or settings.dataset_dir / "messages.csv")
    output_path = Path(args.output or settings.dataset_dir / "output.csv")
    router = _router(args.offline)
    print(f"Routing {input_path} in {router.mode} mode...")
    decisions = route_file(router, input_path, output_path, use_cache=not args.no_cache)
    actions = Counter(d.action.value for d in decisions)
    sources = Counter(d.source.value for d in decisions)
    print(f"\nWrote {len(decisions)} decisions to {output_path}")
    print("Actions: " + ", ".join(f"{k}={v}" for k, v in actions.most_common()))
    print("Decided by: " + ", ".join(f"{k}={v}" for k, v in sources.most_common()))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from notif_router.evaluation import evaluate

    settings = get_settings()
    labelled = Path(args.input or settings.dataset_dir / "sample_messages.csv")
    router = _router(args.offline)
    print(f"Evaluating on {labelled} in {router.mode} mode...\n")
    print(evaluate(router, labelled, use_cache=not args.no_cache).render())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("notif_router.api.app:create_app", factory=True, host=args.host, port=args.port,
                log_level=get_settings().log_level.lower())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="notif-router", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    route = sub.add_parser("route", help="Route a CSV of messages and write predictions")
    route.add_argument("--input", help="Messages CSV (default: dataset/messages.csv)")
    route.add_argument("--output", help="Output CSV (default: dataset/output.csv)")
    route.set_defaults(handler=cmd_route)

    evaluate = sub.add_parser("evaluate", help="Score against labelled sample messages")
    evaluate.add_argument("--input", help="Labelled CSV (default: dataset/sample_messages.csv)")
    evaluate.set_defaults(handler=cmd_evaluate)

    for command in (route, evaluate):
        command.add_argument("--offline", action="store_true",
                             help="Skip the LLM agent and use guardrails + heuristics only")
        command.add_argument("--no-cache", action="store_true",
                             help="Ignore cached agent decisions and re-run the agent")

    serve = sub.add_parser("serve", help="Run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging(get_settings().log_level)
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
