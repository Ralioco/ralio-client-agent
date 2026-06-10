"""Minimal CLI-driven reference client agent for Ralio.

This sample is intentionally independent from any platform SDK or CLI internals.
The only integration point is a constrained command runner. Platform-specific
behavior is supplied as external instructions or skill sources.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from dotenv import load_dotenv


class AgentError(Exception):
    """Base exception for the minimal agent."""


class CommandError(AgentError):
    """Raised when a CLI command cannot be executed safely."""


class ModelError(AgentError):
    """Raised when the model client returns an unusable response."""


DEFAULT_COMMAND_TIMEOUT_SECONDS = 180
DEFAULT_MAX_OUTPUT_CHARS = 12_000
DEFAULT_MAX_TOOL_ROUNDS = 8
DEFAULT_RALIO_SKILL_URL = "https://console.ralio.co/skill.md"
DEFAULT_SKILL_URL_TIMEOUT_SECONDS = 10
DEFAULT_MAX_SKILL_CHARS = 200_000
MAX_COMMAND_TIMEOUT_SECONDS = 600
TRUNCATION_MARKER = "\n[output truncated]"
COMMAND_ENV_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "RALIO_API_URL",
    "RALIO_CONSOLE_URL",
    "TERM",
    "TMP",
    "TMPDIR",
    "TEMP",
    "USER",
)


@dataclass(frozen=True)
class ToolCall:
    """A model-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    """One model turn: optional final text plus optional tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class ModelClient(ABC):
    """Interface for an LLM client used by the agent loop."""

    @abstractmethod
    def respond(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        """Return the model's next turn."""


class CommandRunner(ABC):
    """Interface around subprocess execution, used for tests."""

    @abstractmethod
    def run(
        self,
        command: list[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        """Run *command* and return the completed process."""


class SubprocessCommandRunner(CommandRunner):
    """Command runner backed by ``subprocess.run`` without a shell."""

    def run(
        self,
        command: list[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )


@dataclass(frozen=True)
class CliCommandResult:
    """Result returned by an allowed CLI command."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass
class CliCommandTool:
    """Small adapter around allowed external CLI commands."""

    allowed_commands: tuple[str, ...]
    timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    runner: CommandRunner = field(default_factory=SubprocessCommandRunner)

    def run(
        self,
        command: Sequence[str],
        timeout_seconds: int | None = None,
    ) -> CliCommandResult:
        """Run one allowed command and return captured stdout/stderr."""
        normalized = self._normalize_command(command)
        executable = normalized[0]
        if executable not in self.allowed_commands:
            allowed = ", ".join(self.allowed_commands) or "(none configured)"
            raise CommandError(
                f"Command {executable!r} is not allowed. Allowed commands: {allowed}."
            )

        resolved_timeout = self._resolve_timeout(timeout_seconds)

        try:
            completed = self.runner.run(
                normalized,
                env=_command_environment(),
                timeout_seconds=resolved_timeout,
            )
        except FileNotFoundError as exc:
            raise CommandError(
                f"Could not find CLI command {executable!r}. Install it or put it on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CommandError(f"Command {executable!r} timed out.") from exc

        stdout, stdout_truncated = _truncate_text(
            completed.stdout,
            self.max_output_chars,
        )
        stderr, stderr_truncated = _truncate_text(
            completed.stderr,
            self.max_output_chars,
        )
        return CliCommandResult(
            command=normalized,
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _normalize_command(self, command: Sequence[str]) -> list[str]:
        """Validate command argv supplied by the model."""
        if not command:
            raise CommandError("Command cannot be empty.")

        normalized: list[str] = []
        for part in command:
            if not isinstance(part, str):
                raise CommandError("Every command argument must be a string.")
            if not part:
                raise CommandError("Command arguments cannot be empty strings.")
            if "\x00" in part:
                raise CommandError("Command arguments cannot contain NUL bytes.")
            normalized.append(part)

        return normalized

    def _resolve_timeout(self, timeout_seconds: int | None) -> int:
        """Resolve and validate the timeout for one CLI command."""
        resolved_timeout = (
            self.timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if isinstance(resolved_timeout, bool) or not isinstance(resolved_timeout, int):
            raise CommandError("Command timeout must be an integer.")
        if resolved_timeout < 1 or resolved_timeout > MAX_COMMAND_TIMEOUT_SECONDS:
            raise CommandError(
                "Command timeout must be between 1 and "
                f"{MAX_COMMAND_TIMEOUT_SECONDS} seconds."
            )
        return resolved_timeout


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Trim large command output before sending it back to the model."""
    if len(text) <= max_chars:
        return text, False
    marker_length = len(TRUNCATION_MARKER)
    if max_chars <= marker_length:
        return text[:max_chars], True
    return text[: max_chars - marker_length] + TRUNCATION_MARKER, True


def _command_environment() -> dict[str, str]:
    """Return the minimal env needed by CLI commands, without broad secrets."""
    return {
        key: value
        for key, value in os.environ.items()
        if key in COMMAND_ENV_ALLOWLIST or key.startswith("LC_")
    }


class OpenAIResponsesModelClient(ModelClient):
    """OpenAI Responses API adapter with function-call extraction."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI, OpenAIError  # noqa: PLC0415
            except ImportError as exc:
                raise ModelError(
                    "The openai package is not installed. Install this project's "
                    "dependencies first."
                ) from exc
            try:
                self._client = OpenAI()
            except OpenAIError as exc:
                raise ModelError(str(exc)) from exc
        return self._client

    def respond(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        try:
            response = self._get_client().responses.create(**payload)
        except Exception as exc:
            raise ModelError(f"Model request failed: {exc}") from exc

        text = getattr(response, "output_text", None) or ""
        tool_calls: list[ToolCall] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            tool_calls.append(
                ToolCall(
                    id=str(getattr(item, "call_id", "") or getattr(item, "id", "")),
                    name=str(getattr(item, "name", "")),
                    arguments=_parse_tool_arguments(
                        str(getattr(item, "arguments", "{}") or "{}")
                    ),
                )
            )

        return ModelTurn(text=text, tool_calls=tuple(tool_calls))


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse model-supplied function arguments into an object."""
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ModelError(
            f"Tool arguments were not valid JSON: {raw_arguments}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelError("Tool arguments must decode to a JSON object.")
    return parsed


def cli_command_tool_schema(allowed_commands: tuple[str, ...]) -> dict[str, Any]:
    """Build the command tool schema with configured allowlist context."""
    allowed = ", ".join(allowed_commands) or "(none configured)"
    return {
        "type": "function",
        "name": "run_cli_command",
        "description": (
            "Run one allowed CLI command as an argv array. No shell is used, so "
            "pipes, redirects, glob expansion, environment assignments, and shell "
            "operators are not available. Use command help output and any supplied "
            f"skill instructions to decide which commands to run. Allowed executable "
            f"names: {allowed}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "The exact command argv to run, for example "
                        '["some-cli", "--json", "status"].'
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_COMMAND_TIMEOUT_SECONDS,
                    "description": "Optional timeout for this command.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    }


DEFAULT_INSTRUCTIONS = """\
You are a compact CLI automation assistant.

You do not have built-in knowledge of any product, platform, account, or
payment system. Use supplied skill instructions and command output as the source
of truth. Do not invent external state.

When a user asks for information or action that depends on an external system,
use run_cli_command with an allowed executable. Inspect command help if the
supplied skills are insufficient. Commands must be passed as argv arrays; do not
write shell syntax.

If a command returns a nonzero exit code, read stdout and stderr, then either
recover with another command or explain the concrete failure. If the required
command or skill is not available, say exactly what is missing.
"""


DEFAULT_OPENAI_MODEL = "gpt-5.4"


@dataclass(frozen=True)
class AgentLoopConfig:
    """Configuration for the minimal agent loop."""

    instructions: str = DEFAULT_INSTRUCTIONS
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS


class MinimalCliAgent:
    """A minimal ReAct-style loop with one generic CLI command tool."""

    def __init__(
        self,
        *,
        model: ModelClient,
        cli: CliCommandTool,
        config: AgentLoopConfig | None = None,
        session_id: str | None = None,
        skill_texts: Sequence[str] = (),
    ) -> None:
        self.model = model
        self.cli = cli
        self.config = config or AgentLoopConfig()
        self.session_id = session_id or str(uuid.uuid4())
        self.skill_texts = tuple(skill_texts)
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Start a fresh model conversation and generic CLI session id."""
        self.messages = []
        self.session_id = str(uuid.uuid4())

    def run(self, user_request: str) -> str:
        """Run the agent until it returns a final answer."""
        if not user_request.strip():
            raise AgentError("Empty user request.")

        self.messages.append({"role": "user", "content": user_request})
        tools = [cli_command_tool_schema(self.cli.allowed_commands)]

        for _round_num in range(self.config.max_tool_rounds):
            turn = self.model.respond(
                instructions=self._instructions(),
                messages=self.messages,
                tools=tools,
            )

            if turn.text:
                self.messages.append({"role": "assistant", "content": turn.text})

            if not turn.tool_calls:
                if turn.text:
                    return turn.text
                raise ModelError("Model returned no text and no tool calls.")

            for call in turn.tool_calls:
                self.messages.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                )
                output = self._execute_tool_call(call)
                self.messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.id,
                        "output": json.dumps(output),
                    }
                )

        raise AgentError(
            f"Agent exceeded {self.config.max_tool_rounds} tool rounds without "
            "a final answer."
        )

    def _instructions(self) -> str:
        """Build per-turn instructions, including the current session id."""
        parts = [
            self.config.instructions,
            (
                "Current CLI session id: "
                f"{self.session_id}\n"
                "If a supplied skill says a command needs a stable session, "
                "conversation, or correlation id, use this value."
            ),
        ]
        if self.skill_texts:
            skill_block = "\n\n---\n\n".join(self.skill_texts)
            parts.append("Supplied skill instructions:\n\n" + skill_block)
        return "\n\n".join(parts)

    def _execute_tool_call(self, call: ToolCall) -> dict[str, Any]:
        """Execute one supported tool call and return JSON-serializable output."""
        if call.name != "run_cli_command":
            raise ModelError(f"Unknown tool requested by model: {call.name}")

        command = call.arguments.get("command")
        if not isinstance(command, list):
            return {"error": "run_cli_command requires a `command` array."}

        timeout_seconds = call.arguments.get("timeout_seconds")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int)
        ):
            return {"error": "run_cli_command `timeout_seconds` must be an integer."}

        try:
            result = self.cli.run(command, timeout_seconds=timeout_seconds)
        except CommandError as exc:
            return {"error": str(exc)}

        return {
            "command": result.command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
        }


def build_agent_from_args(args: argparse.Namespace) -> MinimalCliAgent:
    """Create a configured agent from CLI args and environment variables."""
    model_name = args.model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    max_tool_rounds = _int_setting(
        args.max_tool_rounds,
        arg_name="--max-tool-rounds",
        env_key="AGENT_MAX_TOOL_ROUNDS",
        default=DEFAULT_MAX_TOOL_ROUNDS,
    )
    timeout_seconds = _int_setting(
        args.timeout_seconds,
        arg_name="--timeout-seconds",
        env_key="CLI_AGENT_TIMEOUT_SECONDS",
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        max_value=MAX_COMMAND_TIMEOUT_SECONDS,
    )
    max_output_chars = _int_setting(
        args.max_output_chars,
        arg_name="--max-output-chars",
        env_key="CLI_AGENT_MAX_OUTPUT_CHARS",
        default=DEFAULT_MAX_OUTPUT_CHARS,
    )
    allowed_commands = _allowed_commands_from_args(args)
    skill_texts = _load_skills(
        skill_files=args.skill_file or [],
        skill_urls=_skill_urls_from_args(args, allowed_commands),
    )

    cli = CliCommandTool(
        allowed_commands=allowed_commands,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    return MinimalCliAgent(
        model=OpenAIResponsesModelClient(model_name),
        cli=cli,
        config=AgentLoopConfig(max_tool_rounds=max_tool_rounds),
        session_id=args.session_id,
        skill_texts=skill_texts,
    )


def _int_setting(
    arg_value: int | None,
    *,
    arg_name: str,
    env_key: str,
    default: int,
    min_value: int = 1,
    max_value: int | None = None,
) -> int:
    """Resolve one integer setting from CLI args, env, or a default."""
    label = arg_name
    if arg_value is None:
        raw_env_value = os.getenv(env_key)
        if raw_env_value is None or raw_env_value.strip() == "":
            value = default
        else:
            label = env_key
            try:
                value = int(raw_env_value)
            except ValueError as exc:
                raise AgentError(f"{env_key} must be an integer.") from exc
    else:
        value = arg_value

    if value < min_value:
        raise AgentError(f"{label} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise AgentError(f"{label} must be at most {max_value}.")
    return value


def _allowed_commands_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Resolve allowed executable names from args and environment."""
    commands: list[str] = []
    env_value = os.getenv("CLI_AGENT_ALLOWED_COMMANDS", "")
    for command in env_value.split(","):
        stripped = command.strip()
        if stripped:
            commands.append(stripped)
    for command in args.allow_command or []:
        stripped = command.strip()
        if stripped:
            commands.append(stripped)
    return tuple(dict.fromkeys(commands))


def _skill_urls_from_args(
    args: argparse.Namespace,
    allowed_commands: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve external skill URLs from args and Ralio defaults."""
    urls: list[str] = []
    if "ralio" in allowed_commands and not args.no_default_ralio_skill:
        urls.append(DEFAULT_RALIO_SKILL_URL)
    urls.extend(args.skill_url or [])
    return tuple(dict.fromkeys(urls))


def _load_skills(
    *,
    skill_files: Sequence[str],
    skill_urls: Sequence[str],
) -> tuple[str, ...]:
    """Read external skill sources to append to the model instructions."""
    skill_texts: list[str] = []
    for raw_path in skill_files:
        path = Path(raw_path).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AgentError(f"Could not read skill file {raw_path!r}: {exc}") from exc
        skill_texts.append(f"# Skill file: {path}\n\n{text.strip()}")
    for raw_url in skill_urls:
        skill_texts.append(_load_skill_url(raw_url))
    return tuple(skill_texts)


def _load_skill_url(raw_url: str) -> str:
    """Fetch one HTTPS skill URL to append to the model instructions."""
    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise AgentError(f"Skill URL must be an HTTPS URL: {raw_url!r}")

    try:
        with urllib.request.urlopen(  # noqa: S310 - user/default HTTPS skill URL.
            raw_url,
            timeout=DEFAULT_SKILL_URL_TIMEOUT_SECONDS,
        ) as response:
            status = getattr(response, "status", 200)
            if status >= 400:
                raise AgentError(f"Could not fetch skill URL {raw_url!r}: HTTP {status}")
            content = response.read(DEFAULT_MAX_SKILL_CHARS + 1)
    except AgentError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise AgentError(f"Could not fetch skill URL {raw_url!r}: {exc}") from exc

    if len(content) > DEFAULT_MAX_SKILL_CHARS:
        raise AgentError(
            f"Skill URL {raw_url!r} is larger than {DEFAULT_MAX_SKILL_CHARS} bytes."
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentError(f"Skill URL {raw_url!r} did not return UTF-8 text.") from exc

    return f"# Skill URL: {raw_url}\n\n{text.strip()}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Ralio client agent.")
    parser.add_argument(
        "instruction",
        nargs="*",
        help=(
            "Optional first user request. Terminal runs stay interactive by "
            "default; piped stdin is read once."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the initial request or piped stdin once and exit.",
    )
    parser.add_argument(
        "--allow-command",
        action="append",
        help=(
            "Executable name the agent may run. Can be repeated. Also supports "
            "CLI_AGENT_ALLOWED_COMMANDS as a comma-separated env var."
        ),
    )
    parser.add_argument(
        "--skill-file",
        action="append",
        help="Markdown/text skill file to append to the model instructions.",
    )
    parser.add_argument(
        "--skill-url",
        action="append",
        help="HTTPS Markdown/text skill URL to append to the model instructions.",
    )
    parser.add_argument(
        "--no-default-ralio-skill",
        action="store_true",
        help=(
            "Do not automatically fetch the hosted Ralio skill when `ralio` is "
            "allowed."
        ),
    )
    parser.add_argument(
        "--session-id",
        help=("Stable generic session id exposed to skills. Defaults to a new UUID."),
    )
    parser.add_argument(
        "--model",
        help=f"OpenAI model. Defaults to OPENAI_MODEL or {DEFAULT_OPENAI_MODEL}.",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        help="Maximum model/tool loop rounds. Defaults to AGENT_MAX_TOOL_ROUNDS or 8.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Default timeout for each CLI call.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        help="Maximum stdout/stderr characters returned to the model per stream.",
    )
    return parser.parse_args(argv)


def _initial_instruction_from_args(args: argparse.Namespace) -> str | None:
    """Resolve the optional first instruction from argv or stdin."""
    if args.instruction:
        return " ".join(args.instruction).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


def _run_interactive(
    agent: MinimalCliAgent,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    error_stream: TextIO = sys.stderr,
) -> int:
    """Run a simple terminal REPL until EOF or an exit command."""
    is_terminal = bool(getattr(input_stream, "isatty", lambda: False)())
    if is_terminal:
        output_stream.write(
            "Interactive CLI agent. Type /quit or /exit to close, "
            "/new for a fresh session, /id for the session id.\n"
        )
        output_stream.flush()

    while True:
        try:
            if is_terminal:
                output_stream.write("You> ")
                output_stream.flush()
            line = input_stream.readline()
        except KeyboardInterrupt:
            output_stream.write("\n")
            return 0

        if line == "":
            return 0

        user_request = line.strip()
        if not user_request:
            continue

        command = user_request.lower()
        if command in {"/quit", "/exit"}:
            return 0
        if command == "/new":
            agent.reset()
            output_stream.write(f"Started new CLI session: {agent.session_id}\n")
            output_stream.flush()
            continue
        if command == "/id":
            output_stream.write(f"{agent.session_id}\n")
            output_stream.flush()
            continue

        try:
            output_stream.write(agent.run(user_request) + "\n")
            output_stream.flush()
        except AgentError as exc:
            error_stream.write(f"Error: {exc}\n")
            error_stream.flush()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    load_dotenv()
    args = parse_args(argv or sys.argv[1:])
    should_open_repl = not args.once and sys.stdin.isatty()
    try:
        agent = build_agent_from_args(args)
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        instruction = _initial_instruction_from_args(args)
        if args.once and instruction is None:
            print(
                "Error: --once requires an instruction or piped stdin.", file=sys.stderr
            )
            return 1
        if instruction is not None:
            print(agent.run(instruction))
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if not should_open_repl:
            return 1

    if should_open_repl:
        return _run_interactive(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
