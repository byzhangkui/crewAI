"""CLI-based LLM provider for CrewAI.

This module implements a BaseLLM subclass that routes LLM calls
to CLI tools (claude, codex, gemini) via subprocess instead of API calls.

Supported CLI backends:
    - claude: Claude Code CLI (`claude -p "prompt"`)
    - codex: OpenAI Codex CLI (`codex exec "prompt"`)
    - gemini: Gemini CLI (`gemini -p "prompt"`)

Usage:
    from crewai import LLM
    llm = LLM(model="cli/claude")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from crewai.llms.base_llm import BaseLLM, get_current_call_id, llm_call_context
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)

if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


logger = logging.getLogger(__name__)


# CLI backend configurations
CLI_BACKENDS: dict[str, dict[str, Any]] = {
    "claude": {
        "cmd": "claude",
        "args_template": ["-p", "{prompt}"],
        "env_check": None,
        "supports_system_prompt": True,
        "system_prompt_args": ["--system-prompt", "{system}"],
    },
    "codex": {
        "cmd": "codex",
        "args_template": ["exec", "{prompt}"],
        "env_check": None,
        "supports_system_prompt": False,
        "system_prompt_args": [],
    },
    "gemini": {
        "cmd": "gemini",
        "args_template": ["-p", "{prompt}"],
        "env_check": None,
        "supports_system_prompt": True,
        "system_prompt_args": ["--system", "{system}"],
    },
}


class CliCompletion(BaseLLM):
    """LLM implementation that calls CLI tools via subprocess.

    This allows CrewAI to use CLI-authenticated LLM tools
    (claude, codex, gemini) without requiring API keys.
    """

    provider: str = "cli"
    cli_backend: str = "claude"
    cli_cmd: str | None = None
    cli_timeout: int = Field(default=300, description="CLI call timeout in seconds")
    verbose: bool = False

    @model_validator(mode="before")
    @classmethod
    def _validate_cli_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        model = data.get("model", "")

        # Extract backend from model name (e.g., "claude" from "cli/claude")
        if "/" in model:
            _, _, backend = model.partition("/")
            # Handle sub-paths like "claude-sonnet" -> use "claude" backend
            base_backend = backend.split("-")[0].split("/")[0]
            if base_backend in CLI_BACKENDS:
                data["cli_backend"] = base_backend
            else:
                data["cli_backend"] = backend
        elif model in CLI_BACKENDS:
            data["cli_backend"] = model

        data["provider"] = "cli"
        return data

    @model_validator(mode="after")
    def _validate_cli_available(self) -> CliCompletion:
        """Check that the CLI tool is actually installed."""
        backend = self.cli_backend
        config = CLI_BACKENDS.get(backend)
        if not config:
            raise ValueError(
                f"Unknown CLI backend: '{backend}'. "
                f"Supported: {', '.join(CLI_BACKENDS.keys())}"
            )

        cmd = self.cli_cmd or config["cmd"]
        if not shutil.which(cmd):
            raise ValueError(
                f"CLI tool '{cmd}' not found in PATH. "
                f"Please install it first."
            )

        return self

    def _format_messages_to_prompt(
        self, messages: str | list[LLMMessage]
    ) -> tuple[str, str | None]:
        """Convert CrewAI messages to a single prompt string.

        Returns:
            Tuple of (user_prompt, system_prompt_or_none)
        """
        if isinstance(messages, str):
            return messages, None

        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle list-type content (multimodal)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item["text"])
                content = "\n".join(text_parts)

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                conversation_parts.append(content)
            elif role == "assistant":
                conversation_parts.append(f"[Previous Assistant Response]\n{content}")
            elif role == "tool":
                conversation_parts.append(f"[Tool Result]\n{content}")

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        user_prompt = "\n\n---\n\n".join(conversation_parts)

        return user_prompt, system_prompt

    def _build_cli_command(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[str]:
        """Build the CLI command arguments."""
        backend = self.cli_backend
        config = CLI_BACKENDS[backend]
        cmd = self.cli_cmd or config["cmd"]

        # Build full prompt: if CLI doesn't support system prompt, prepend it
        full_prompt = prompt
        if system_prompt:
            if config["supports_system_prompt"]:
                args = [cmd]
                for arg in config["system_prompt_args"]:
                    args.append(arg.replace("{system}", system_prompt))
                for arg in config["args_template"]:
                    args.append(arg.replace("{prompt}", full_prompt))
                return args
            else:
                # Prepend system prompt to user prompt
                full_prompt = f"[System Instructions]\n{system_prompt}\n\n[Task]\n{prompt}"

        args = [cmd]
        for arg in config["args_template"]:
            args.append(arg.replace("{prompt}", full_prompt))
        return args

    def _call_cli(self, prompt: str, system_prompt: str | None = None) -> str:
        """Execute CLI command and return the output."""
        cmd = self._build_cli_command(prompt, system_prompt)

        if self.verbose:
            logger.info(f"[CLI LLM] Calling: {cmd[0]} ...")
            logger.debug(f"[CLI LLM] Full command: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cli_timeout,
                env={**os.environ},
            )

            if result.returncode != 0:
                error = result.stderr.strip() or f"CLI exited with code {result.returncode}"
                raise RuntimeError(f"CLI call failed: {error}")

            output = result.stdout.strip()
            if not output:
                raise RuntimeError("CLI returned empty output")

            return output

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"CLI call timed out after {self.cli_timeout}s. "
                f"Increase cli_timeout if your prompts need more time."
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"CLI tool '{cmd[0]}' not found. Is it installed and in PATH?"
            )

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call the CLI LLM backend.

        Converts messages to a prompt, calls the CLI, and returns the output.
        Tool calling is handled by appending tool descriptions to the prompt
        and parsing structured output from the CLI response.
        """
        with llm_call_context():
            try:
                # Emit start event
                call_id = get_current_call_id()
                crewai_event_bus.emit(
                    self,
                    LLMCallStartedEvent(
                        call_id=call_id,
                        model=self.model,
                        call_type=LLMCallType.LLM_CALL,
                        messages=messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
                        from_task=from_task,
                        from_agent=from_agent,
                    ),
                )

                # Format messages to prompt
                prompt, system_prompt = self._format_messages_to_prompt(messages)

                # If tools are provided, append tool descriptions to prompt
                if tools and available_functions:
                    tool_desc = self._format_tools_for_prompt(tools)
                    prompt = f"{prompt}\n\n{tool_desc}"

                # Call CLI
                output = self._call_cli(prompt, system_prompt)

                # Try to handle tool calls from output
                if tools and available_functions:
                    tool_result = self._try_parse_tool_call(output, available_functions)
                    if tool_result is not None:
                        crewai_event_bus.emit(
                            self,
                            LLMCallCompletedEvent(
                                call_id=call_id,
                                model=self.model,
                                call_type=LLMCallType.TOOL_CALL,
                                response=str(tool_result),
                                from_task=from_task,
                                from_agent=from_agent,
                            ),
                        )
                        return tool_result

                # Handle response_model (structured output)
                if response_model:
                    try:
                        parsed = self._parse_structured_output(output, response_model)
                        crewai_event_bus.emit(
                            self,
                            LLMCallCompletedEvent(
                                call_id=call_id,
                                model=self.model,
                                call_type=LLMCallType.LLM_CALL,
                                response=parsed.model_dump_json() if hasattr(parsed, 'model_dump_json') else str(parsed),
                                from_task=from_task,
                                from_agent=from_agent,
                            ),
                        )
                        return parsed
                    except Exception as e:
                        logger.warning(f"Structured output parsing failed: {e}")

                # Return raw text
                # Track token usage (approximate)
                self._token_usage["successful_requests"] += 1
                self._token_usage["prompt_tokens"] += len(prompt) // 4
                self._token_usage["completion_tokens"] += len(output) // 4
                self._token_usage["total_tokens"] = (
                    self._token_usage["prompt_tokens"]
                    + self._token_usage["completion_tokens"]
                )

                crewai_event_bus.emit(
                    self,
                    LLMCallCompletedEvent(
                        call_id=call_id,
                        model=self.model,
                        call_type=LLMCallType.LLM_CALL,
                        response=output,
                        from_task=from_task,
                        from_agent=from_agent,
                    ),
                )

                return output

            except Exception as e:
                error_msg = f"CLI LLM call failed: {e!s}"
                logger.error(error_msg)
                try:
                    crewai_event_bus.emit(
                        self,
                        LLMCallFailedEvent(
                            call_id=get_current_call_id(),
                            model=self.model,
                            call_type=LLMCallType.LLM_CALL,
                            error=error_msg,
                            from_task=from_task,
                            from_agent=from_agent,
                        ),
                    )
                except Exception:
                    pass
                raise

    def _format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions into a text block for the CLI prompt."""
        lines = [
            "You have access to the following tools. To use a tool, respond with EXACTLY this JSON format:",
            '{"tool_call": {"name": "tool_name", "arguments": {"arg1": "value1"}}}',
            "",
            "Available tools:",
        ]
        for tool in tools:
            if isinstance(tool, dict):
                func = tool.get("function", tool)
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                params = func.get("parameters", {})
                lines.append(f"\n- {name}: {desc}")
                if params.get("properties"):
                    lines.append(f"  Parameters: {json.dumps(params['properties'], ensure_ascii=False)}")
        lines.append("\nIf no tool is needed, just respond with your answer directly.")
        return "\n".join(lines)

    def _try_parse_tool_call(
        self, output: str, available_functions: dict[str, Any]
    ) -> Any | None:
        """Try to parse a tool call from CLI output.

        Returns the tool execution result if a valid tool call is found,
        None otherwise.
        """
        try:
            # Try to find JSON in the output
            import re
            json_match = re.search(r'\{[\s\S]*"tool_call"[\s\S]*\}', output)
            if not json_match:
                return None

            parsed = json.loads(json_match.group())
            tool_call = parsed.get("tool_call")
            if not tool_call:
                return None

            func_name = tool_call.get("name", "")
            func_args = tool_call.get("arguments", {})

            if func_name not in available_functions:
                logger.warning(f"Tool '{func_name}' not found in available functions")
                return None

            # Execute the tool
            func = available_functions[func_name]
            if callable(func):
                return func(**func_args)
            return None

        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _parse_structured_output(
        self, output: str, response_model: type[BaseModel]
    ) -> Any:
        """Parse CLI output into a structured Pydantic model."""
        import re
        # Try to extract JSON from the output
        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return response_model(**data)
            except (json.JSONDecodeError, Exception):
                pass

        # Try the entire output as JSON
        try:
            data = json.loads(output)
            return response_model(**data)
        except (json.JSONDecodeError, Exception):
            raise ValueError(
                f"Could not parse CLI output into {response_model.__name__}. "
                f"Output: {output[:200]}..."
            )

    def supports_function_calling(self) -> bool:
        """CLI backends support function calling via prompt engineering."""
        return True

    def supports_stop_words(self) -> bool:
        """CLI backends don't support stop words."""
        return False

    def get_context_window_size(self) -> int:
        """Return a large context window size for CLI backends."""
        return 200000  # Claude Code supports large contexts
