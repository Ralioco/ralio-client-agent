from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from agent import (
    AgentError,
    AgentLoopConfig,
    CliCommandTool,
    CommandError,
    CommandRunner,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RALIO_SKILL_URL,
    MinimalCliAgent,
    ModelClient,
    ModelError,
    ModelTurn,
    OpenAIResponsesModelClient,
    THINKING_FINISHED,
    THINKING_STARTED,
    ToolCall,
    build_agent_from_args,
    main,
    parse_args,
    _truncate_text,
    _run_interactive,
)


class FakeRunner(CommandRunner):
    def __init__(
        self,
        responses: list[subprocess.CompletedProcess[str]],
        *,
        stream_chunks: list[tuple[str, str]] | None = None,
    ) -> None:
        self.responses = responses
        self.stream_chunks = stream_chunks or []
        self.calls: list[tuple[list[str], Mapping[str, str], int]] = []

    def run(
        self,
        command: list[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: int,
        output_callback: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, env, timeout_seconds))
        if output_callback is not None:
            for stream_name, chunk in self.stream_chunks:
                output_callback(stream_name, chunk)
        if not self.responses:
            raise AssertionError("No fake subprocess response configured.")
        return self.responses.pop(0)


class ScriptedModel(ModelClient):
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = turns
        self.calls: list[dict[str, Any]] = []

    def respond(
        self,
        *,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        self.calls.append(
            {
                "instructions": instructions,
                "messages": [dict(message) for message in messages],
                "tools": tools,
            }
        )
        if not self.turns:
            raise AssertionError("No scripted model turn configured.")
        return self.turns.pop(0)


class FakeSkillUrlResponse:
    status = 200

    def __init__(self, text: str) -> None:
        self.text = text

    def __enter__(self) -> FakeSkillUrlResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        return self.text.encode("utf-8")


def _completed(
    payload: dict[str, Any] | list[Any] | str,
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    stdout = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.CompletedProcess(
        args=["demo-cli"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_cli_command_tool_runs_allowed_command_and_returns_exit_code() -> None:
    runner = FakeRunner(
        [
            _completed(
                {"status": "ok"},
                returncode=7,
                stderr="recoverable warning",
            )
        ]
    )
    tool = CliCommandTool(
        allowed_commands=("demo-cli",),
        timeout_seconds=42,
        max_output_chars=1_000,
        runner=runner,
    )

    result = tool.run(["demo-cli", "--json", "status"])

    command, _env, timeout_seconds = runner.calls[0]
    assert command == ["demo-cli", "--json", "status"]
    assert timeout_seconds == 42
    assert result.command == ["demo-cli", "--json", "status"]
    assert result.returncode == 7
    assert result.stdout == '{"status": "ok"}'
    assert result.stderr == "recoverable warning"


def test_cli_command_tool_reports_approval_wait_progress_once() -> None:
    runner = FakeRunner(
        [_completed("done")],
        stream_chunks=[
            ("stdout", "Awaiting human approval\n"),
            ("stdout", "Awaiting human approval\n"),
        ],
    )
    tool = CliCommandTool(
        allowed_commands=("demo-cli",),
        runner=runner,
    )
    progress_messages: list[str] = []

    result = tool.run(
        ["demo-cli", "chat"],
        progress_callback=progress_messages.append,
    )

    assert result.stdout == "done"
    assert progress_messages == [
        "Status: request submitted and waiting for human approval. "
        "I'll keep waiting for the CLI result."
    ]


def test_cli_command_tool_rejects_unallowed_command() -> None:
    tool = CliCommandTool(
        allowed_commands=("demo-cli",),
        runner=FakeRunner([]),
    )

    with pytest.raises(CommandError, match="not allowed"):
        tool.run(["bash", "-lc", "demo-cli status"])


@pytest.mark.parametrize("timeout_seconds", [0, 601])
def test_cli_command_tool_rejects_out_of_range_timeout(
    timeout_seconds: int,
) -> None:
    runner = FakeRunner([])
    tool = CliCommandTool(
        allowed_commands=("demo-cli",),
        runner=runner,
    )

    with pytest.raises(CommandError, match="between 1 and 600"):
        tool.run(["demo-cli", "--json", "status"], timeout_seconds=timeout_seconds)

    assert runner.calls == []


def test_cli_command_tool_scrubs_openai_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/tmp/ralio-client-agent-home")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("OPENAI_MODEL", "private-model")
    monkeypatch.setenv("RALIO_API_URL", "https://api.ralio.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "supabase-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")
    runner = FakeRunner([_completed({"status": "ok"})])
    tool = CliCommandTool(
        allowed_commands=("demo-cli",),
        runner=runner,
    )

    tool.run(["demo-cli", "--json", "status"])

    _command, env, _timeout_seconds = runner.calls[0]
    assert env["HOME"] == "/tmp/ralio-client-agent-home"
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_MODEL" not in env
    assert "SUPABASE_SECRET_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["RALIO_API_URL"] == "https://api.ralio.co"


def test_truncate_text_respects_hard_max_chars() -> None:
    text, was_truncated = _truncate_text("x" * 100, max_chars=20)

    assert was_truncated is True
    assert len(text) <= 20
    assert text.endswith("[output truncated]")


def test_agent_loop_executes_generic_cli_tool_then_returns_final_text() -> None:
    runner = FakeRunner(
        [
            _completed(
                {
                    "accounts": [
                        {"name": "Main"},
                        {"name": "Savings"},
                    ]
                }
            )
        ]
    )
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="run_cli_command",
                        arguments={
                            "command": ["demo-cli", "--json", "accounts", "list"],
                            "timeout_seconds": 5,
                        },
                    ),
                )
            ),
            ModelTurn(text="The accessible account names are Main and Savings."),
        ]
    )
    agent = MinimalCliAgent(
        model=model,
        cli=CliCommandTool(
            allowed_commands=("demo-cli",),
            runner=runner,
        ),
        config=AgentLoopConfig(max_tool_rounds=2),
        session_id="session-1",
        skill_texts=("Use demo-cli for account questions.",),
    )

    reply = agent.run("List account names only.")

    assert reply == "The accessible account names are Main and Savings."
    assert runner.calls[0][0] == ["demo-cli", "--json", "accounts", "list"]
    assert runner.calls[0][2] == 5
    assert "Current CLI session id: session-1" in model.calls[0]["instructions"]
    assert "Use demo-cli for account questions." in model.calls[0]["instructions"]
    assert (
        "Allowed executable names: demo-cli"
        in model.calls[0]["tools"][0]["description"]
    )
    second_messages = model.calls[1]["messages"]
    assert second_messages[-2]["type"] == "function_call"
    assert second_messages[-2]["name"] == "run_cli_command"
    assert second_messages[-1]["type"] == "function_call_output"
    output = json.loads(second_messages[-1]["output"])
    assert output["command"] == ["demo-cli", "--json", "accounts", "list"]
    assert output["returncode"] == 0
    assert json.loads(output["stdout"])["accounts"][0]["name"] == "Main"


def test_agent_reports_thinking_activity_around_model_calls() -> None:
    runner = FakeRunner([_completed({"status": "ok"})])
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="run_cli_command",
                        arguments={"command": ["demo-cli", "status"]},
                    ),
                )
            ),
            ModelTurn(text="Done."),
        ]
    )
    agent = MinimalCliAgent(
        model=model,
        cli=CliCommandTool(
            allowed_commands=("demo-cli",),
            runner=runner,
        ),
        config=AgentLoopConfig(max_tool_rounds=2),
    )
    events: list[str] = []

    reply = agent.run("Check status.", activity_callback=events.append)

    assert reply == "Done."
    assert events == [
        THINKING_STARTED,
        THINKING_FINISHED,
        THINKING_STARTED,
        THINKING_FINISHED,
    ]


def test_agent_returns_command_errors_to_model_for_recovery() -> None:
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="run_cli_command",
                        arguments={"command": ["bash", "-lc", "demo-cli status"]},
                    ),
                )
            ),
            ModelTurn(text="I could not run that command because it is not allowed."),
        ]
    )
    runner = FakeRunner([])
    agent = MinimalCliAgent(
        model=model,
        cli=CliCommandTool(
            allowed_commands=("demo-cli",),
            runner=runner,
        ),
        config=AgentLoopConfig(max_tool_rounds=2),
    )

    reply = agent.run("Check status.")

    assert reply == "I could not run that command because it is not allowed."
    assert runner.calls == []
    output = json.loads(model.calls[1]["messages"][-1]["output"])
    assert "not allowed" in output["error"]


def test_agent_keeps_history_between_interactive_turns() -> None:
    model = ScriptedModel(
        [
            ModelTurn(text="First reply."),
            ModelTurn(text="Second reply."),
        ]
    )
    agent = MinimalCliAgent(
        model=model,
        cli=CliCommandTool(allowed_commands=("demo-cli",), runner=FakeRunner([])),
    )

    assert agent.run("First question.") == "First reply."
    assert agent.run("Follow-up question.") == "Second reply."

    assert model.calls[0]["messages"] == [
        {"role": "user", "content": "First question."}
    ]
    assert model.calls[1]["messages"] == [
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First reply."},
        {"role": "user", "content": "Follow-up question."},
    ]


def test_interactive_new_command_resets_history_and_session() -> None:
    model = ScriptedModel(
        [
            ModelTurn(text="Before reset."),
            ModelTurn(text="After reset."),
        ]
    )
    agent = MinimalCliAgent(
        model=model,
        cli=CliCommandTool(allowed_commands=("demo-cli",), runner=FakeRunner([])),
        session_id="session-before",
    )
    output = StringIO()
    error = StringIO()

    status = _run_interactive(
        agent,
        input_stream=StringIO("first\n/new\nsecond\n/quit\n"),
        output_stream=output,
        error_stream=error,
    )

    assert status == 0
    assert "Before reset." in output.getvalue()
    assert "Started new CLI session:" in output.getvalue()
    assert "After reset." in output.getvalue()
    assert error.getvalue() == ""
    assert agent.session_id != "session-before"
    assert model.calls[1]["messages"] == [
        {"role": "user", "content": "second"},
    ]


def test_interactive_session_prints_approval_wait_progress() -> None:
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="run_cli_command",
                        arguments={"command": ["demo-cli", "chat"]},
                    ),
                )
            ),
            ModelTurn(text="The request is still pending approval."),
        ]
    )
    runner = FakeRunner(
        [_completed("Approval not completed within the wait window.")],
        stream_chunks=[("stdout", "Awaiting human approval\n")],
    )
    agent = MinimalCliAgent(
        model=model,
        cli=CliCommandTool(allowed_commands=("demo-cli",), runner=runner),
    )
    output = StringIO()

    status = _run_interactive(
        agent,
        input_stream=StringIO("send money\n/quit\n"),
        output_stream=output,
        error_stream=StringIO(),
    )

    transcript = output.getvalue()
    progress_index = transcript.index("waiting for human approval")
    final_index = transcript.index("The request is still pending approval.")
    assert status == 0
    assert progress_index < final_index


def test_build_agent_uses_default_openai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    args = parse_args(["--allow-command", "demo-cli"])

    agent = build_agent_from_args(args)

    assert isinstance(agent.model, OpenAIResponsesModelClient)
    assert agent.model.model == DEFAULT_OPENAI_MODEL
    assert DEFAULT_OPENAI_MODEL == "gpt-5.4-mini"


def test_build_agent_loads_default_ralio_skill_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_urlopen(url: str, *, timeout: int) -> FakeSkillUrlResponse:
        calls.append((url, timeout))
        return FakeSkillUrlResponse("# Hosted Ralio skill")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("agent.SKILL_CACHE_DIR", Path("/nonexistent/cache/dir"))
    args = parse_args(["--allow-command", "ralio"])

    agent = build_agent_from_args(args)

    assert calls == [(DEFAULT_RALIO_SKILL_URL, 10)]
    assert agent.skill_texts == (
        f"# Skill URL: {DEFAULT_RALIO_SKILL_URL}\n\n# Hosted Ralio skill",
    )


def test_build_agent_can_skip_default_ralio_skill_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(url: str, *, timeout: int) -> FakeSkillUrlResponse:
        raise AssertionError(f"Unexpected skill URL fetch: {url} {timeout}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    args = parse_args(["--allow-command", "ralio", "--no-default-ralio-skill"])

    agent = build_agent_from_args(args)

    assert agent.skill_texts == ()


def test_build_agent_loads_explicit_skill_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_urlopen(url: str, *, timeout: int) -> FakeSkillUrlResponse:
        calls.append((url, timeout))
        return FakeSkillUrlResponse("Use demo-cli carefully.")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("agent.SKILL_CACHE_DIR", Path("/nonexistent/cache/dir"))
    args = parse_args(
        [
            "--allow-command",
            "demo-cli",
            "--skill-url",
            "https://example.com/skill.md",
        ]
    )

    agent = build_agent_from_args(args)

    assert calls == [("https://example.com/skill.md", 10)]
    assert agent.skill_texts == (
        "# Skill URL: https://example.com/skill.md\n\nUse demo-cli carefully.",
    )


def test_build_agent_rejects_non_https_skill_url() -> None:
    args = parse_args(
        [
            "--allow-command",
            "demo-cli",
            "--skill-url",
            "http://example.com/skill.md",
        ]
    )

    with pytest.raises(AgentError, match="HTTPS URL"):
        build_agent_from_args(args)


def test_build_agent_rejects_malformed_numeric_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_MAX_TOOL_ROUNDS", "not-a-number")
    args = parse_args(["--allow-command", "demo-cli"])

    with pytest.raises(AgentError, match="AGENT_MAX_TOOL_ROUNDS must be an integer"):
        build_agent_from_args(args)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--allow-command", "demo-cli", "--max-tool-rounds", "0"],
            "--max-tool-rounds must be at least 1",
        ),
        (
            ["--allow-command", "demo-cli", "--timeout-seconds", "601"],
            "--timeout-seconds must be at most 600",
        ),
    ],
)
def test_build_agent_rejects_out_of_range_numeric_args(
    argv: list[str],
    message: str,
) -> None:
    args = parse_args(argv)

    with pytest.raises(AgentError, match=message):
        build_agent_from_args(args)


def test_openai_client_missing_api_key_raises_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIResponsesModelClient(DEFAULT_OPENAI_MODEL)

    with pytest.raises(ModelError, match="api_key"):
        client._get_client()


def test_main_calls_load_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_load_dotenv(*args: Any, **kwargs: Any) -> bool:
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr("agent.load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(
        "agent.build_agent_from_args",
        lambda _args: None,
    )
    monkeypatch.setattr(
        "agent._initial_instruction_from_args",
        lambda _args: None,
    )

    main(["--allow-command", "demo-cli"])

    assert len(calls) == 1
    assert calls[0] == ((), {})
