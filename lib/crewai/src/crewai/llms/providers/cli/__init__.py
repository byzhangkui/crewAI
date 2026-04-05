"""CLI-based LLM provider for CrewAI.

Routes LLM calls to CLI tools (claude, codex, gemini) via subprocess,
enabling CrewAI workflows without API keys.

Usage:
    llm = LLM(model="cli/claude")       # Uses Claude Code CLI
    llm = LLM(model="cli/codex")        # Uses Codex CLI
    llm = LLM(model="cli/gemini")       # Uses Gemini CLI
"""

from crewai.llms.providers.cli.completion import CliCompletion

__all__ = ["CliCompletion"]
