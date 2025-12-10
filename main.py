"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_cli(config_path: str = "config.yaml"):
    """Run CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main(["--config", config_path])


def run_web():
    """Run web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation(config_path: str = "config.yaml"):
    """Run system evaluation with SystemEvaluator."""
    import yaml
    from dotenv import load_dotenv
    from src.langgraph_orchestrator import LangGraphOrchestrator
    from src.evaluation.evaluator import SystemEvaluator

    # Load environment variables
    load_dotenv()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize AutoGen orchestrator
    print("Initializing AutoGen orchestrator...")
    orchestrator = LangGraphOrchestrator(config)

    # Run full evaluation
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)
    report = await evaluator.evaluate_system("data/example_queries.json")

    summary = report.get("summary", {})
    scores = report.get("scores", {})

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Queries: {summary.get('total_queries', 0)}")
    print(f"Successful: {summary.get('successful', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0.0):.2%}")
    print(f"Overall Average Score: {scores.get('overall_average', 0.0):.3f}")
    if scores.get("by_criterion"):
        print("\nScores by Criterion:")
        for criterion, score in scores["by_criterion"].items():
            print(f"  - {criterion}: {score:.3f}")


def run_autogen():
    """Run AutoGen example."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen"],
        default="autogen",
        help="Mode to run: cli, web, evaluate, or autogen (default)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(args.config)
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation(args.config))
    elif args.mode == "autogen":
        run_autogen()


if __name__ == "__main__":
    main()
