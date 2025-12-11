"""
Example: Using LangGraph for Multi-Agent Research

This script demonstrates how to use the LangGraph-based multi-agent research system.

Usage:
    python example_langgraph.py
"""

import os
import yaml
import logging
from dotenv import load_dotenv
from src.langgraph_orchestrator import LangGraphOrchestrator


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/example.log")
        ]
    )


def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'=' * 70}")
        print(f"{title:^70}")
        print(f"{'=' * 70}\n")
    else:
        print(f"{'=' * 70}\n")


def run_single_query():
    """
    Example 1: Run a single research query.

    This is the simplest way to use the system.
    """
    print_separator("Example 1: Single Research Query")

    # Load environment and config
    load_dotenv()
    config = load_config()

    # Create orchestrator
    orchestrator = LangGraphOrchestrator(config)

    # Define your research query
    query = "What are the key principles of accessible user interface design?"

    print(f"Query: {query}\n")
    print("Processing... (this may take 1-2 minutes)\n")

    # Process the query
    result = orchestrator.process_query(query, max_rounds=20)

    # Display results
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print_separator("Final Response")
    print(result["response"])

    print_separator("Metadata")
    print(f"Messages exchanged: {result['metadata']['num_messages']}")
    print(f"Sources gathered: {result['metadata']['num_sources']}")
    print(f"Agents involved: {', '.join(result['metadata']['agents_involved'])}")


def run_multiple_queries():
    """
    Example 2: Process multiple queries in sequence.

    Shows how to reuse the orchestrator for multiple queries.
    """
    print_separator("Example 2: Multiple Research Queries")

    load_dotenv()
    config = load_config()

    # Create orchestrator once
    orchestrator = LangGraphOrchestrator(config)

    # List of queries to process
    queries = [
        "What is cognitive load theory in HCI?",
        "How does eye tracking improve user experience?",
        "What are best practices for mobile UI design?",
    ]

    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}] {query}")
        print("-" * 70)

        result = orchestrator.process_query(query, max_rounds=15)
        results.append(result)

        # Print brief summary
        if "error" not in result:
            response_preview = result["response"][:200] + "..."
            print(f"Response preview: {response_preview}\n")

    print_separator("Summary")
    print(f"Processed {len(queries)} queries successfully")

    return results


def inspect_conversation():
    """
    Example 3: Inspect the conversation history.

    Shows how to access and examine the agent-to-agent conversation.
    """
    print_separator("Example 3: Inspecting Conversation History")

    load_dotenv()
    config = load_config()

    orchestrator = LangGraphOrchestrator(config)

    query = "What is the difference between usability and user experience?"

    print(f"Query: {query}\n")
    result = orchestrator.process_query(query, max_rounds=20)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print_separator("Conversation Flow")

    # Display each message in the conversation
    for i, msg in enumerate(result["conversation_history"], 1):
        agent = msg.get("source", msg.get("name", "Unknown"))
        content = msg.get("content", "")

        # Truncate long messages for readability
        if len(content) > 300:
            content = content[:300] + "...[truncated]"

        print(f"[{i}] {agent}:")
        print(f"    {content}\n")


def check_setup():
    """
    Check if the system is properly configured.

    Verifies API keys and dependencies.
    """
    print_separator("Setup Check")

    load_dotenv()

    checks = {
        "Environment file (.env)": os.path.exists(".env"),
        "Config file (config.yaml)": os.path.exists("config.yaml"),
        "Logs directory": os.path.exists("logs"),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
    }

    for item, status in checks.items():
        status_str = "OK" if status else "MISSING"
        print(f"{item}: {status_str}")


if __name__ == "__main__":
    setup_logging()
    run_single_query()
