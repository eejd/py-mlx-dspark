"""Agent integration config — model-free (pure config generation)."""

import json

from mlx_dspark import integrations as I


def rows(base="http://127.0.0.1:8080", model="Qwen3-8B-8bit", key=None):
    return I.integrations(base, model, key)


class TestCoverage:
    def test_covers_the_agents_the_readme_names(self):
        ids = {r["id"] for r in rows()}
        assert {"claude-code", "codex", "opencode", "pi", "openai-compatible"} <= ids

    def test_every_row_has_at_least_one_setup_step(self):
        for r in rows():
            assert r["setup"], f"{r['id']} has no setup steps"

    def test_protocols_match_the_documented_setups(self):
        by_id = {r["id"]: r for r in rows()}
        for anthropic in ("claude-code", "pi"):         # pi: the README's tested form
            assert by_id[anthropic]["protocol"] == "anthropic"
        for other in ("codex", "opencode", "openai-compatible"):
            assert by_id[other]["protocol"] == "openai"


class TestClaudeCode:
    def test_env_sets_base_url_auth_and_the_haiku_slot(self):
        env = I.anthropic_env("http://127.0.0.1:8080", "Qwen3-8B-8bit", None)
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
        # Without an auth token Claude Code keeps billing the saved login — must always be set.
        assert env["ANTHROPIC_AUTH_TOKEN"] == "mlx-dspark"
        # The haiku slot is where Claude Code sends background tasks; unmapped, they ask for a
        # model this server never heard of.
        assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "Qwen3-8B-8bit"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "Qwen3-8B-8bit"

    def test_api_key_flows_into_the_token(self):
        env = I.anthropic_env("http://x", "m", "secret")
        assert env["ANTHROPIC_AUTH_TOKEN"] == "secret"

    def test_command_is_bare_for_the_default_url(self):
        cc = next(r for r in rows() if r["id"] == "claude-code")
        command = cc["setup"][0]["command"]
        assert command == "mlx-dspark claude"

    def test_command_carries_url_and_key_when_non_default(self):
        cc = next(r for r in rows(base="http://192.168.1.9:9000", key="abc")
                  if r["id"] == "claude-code")
        command = cc["setup"][0]["command"]
        assert "--url http://192.168.1.9:9000" in command
        assert "--api-key abc" in command


class TestOpenAIAgents:
    def test_all_openai_agents_use_the_v1_base(self):
        for r in rows():
            if r["protocol"] != "openai":
                continue
            blob = json.dumps(r["setup"])
            # the /v1 suffix is the single most common thing to get wrong
            assert "http://127.0.0.1:8080/v1" in blob, f"{r['id']} missing /v1 base"

    def test_codex_toml_is_valid_and_names_the_model(self):
        codex = next(r for r in rows() if r["id"] == "codex")
        toml = codex["setup"][0]["content"]
        assert "[model_providers.mlx-dspark]" in toml
        assert 'model = "Qwen3-8B-8bit"' in toml
        assert codex["setup"][0]["path"] == "~/.codex/config.toml"

    def test_codex_uses_the_responses_api_and_sets_its_key(self):
        """Codex dropped Chat Completions — a ``wire_api = "chat"`` provider cannot connect —
        and fails without the ``env_key`` variable set."""
        import tomllib

        codex = next(r for r in rows(key="abc") if r["id"] == "codex")
        cfg = tomllib.loads(codex["setup"][0]["content"])
        provider = cfg["model_providers"]["mlx-dspark"]
        assert provider["wire_api"] == "responses"
        assert provider["base_url"] == "http://127.0.0.1:8080/v1"
        assert cfg["profiles"]["mlx-dspark"] == {"model_provider": "mlx-dspark",
                                                 "model": "Qwen3-8B-8bit"}
        command = codex["setup"][1]["command"]
        assert f"{provider['env_key']}=abc" in command and "--profile mlx-dspark" in command

    def test_opencode_and_pi_emit_valid_json(self):
        for agent_id in ("opencode", "pi"):
            agent = next(r for r in rows() if r["id"] == agent_id)
            content = agent["setup"][0]["content"]
            parsed = json.loads(content)              # must parse
            assert "Qwen3-8B-8bit" in json.dumps(parsed)

    def test_pi_matches_the_readme_tested_config(self):
        """``~/.pi/agent/models.json``, camelCase keys, the Anthropic endpoint at the bare
        base URL — the form the README measured (not ``~/.pi/models.json`` / ``base_url``)."""
        pi = next(r for r in I.integrations("http://127.0.0.1:8080", "Qwen3-8B-8bit", None,
                                            40960) if r["id"] == "pi")
        step = pi["setup"][0]
        assert step["path"] == "~/.pi/agent/models.json"
        provider = json.loads(step["content"])["providers"]["mlx-dspark"]
        assert provider == {
            "baseUrl": "http://127.0.0.1:8080", "api": "anthropic-messages",
            "apiKey": "mlx-dspark",
            "models": [{"id": "Qwen3-8B-8bit", "contextWindow": 40960, "maxTokens": 8192}],
        }
        assert pi["setup"][1]["command"] == "pi --provider mlx-dspark --model Qwen3-8B-8bit"
        # unknown context window: the limits are left to pi's defaults
        unknown = next(r for r in rows() if r["id"] == "pi")
        models = json.loads(unknown["setup"][0]["content"])["providers"]["mlx-dspark"]["models"]
        assert models == [{"id": "Qwen3-8B-8bit"}]

    def test_openai_compatible_exposes_the_three_fields_a_gui_asks_for(self):
        generic = next(r for r in rows() if r["id"] == "openai-compatible")
        fields = generic["setup"][0]["fields"]
        assert fields["Base URL"] == "http://127.0.0.1:8080/v1"
        assert fields["Model"] == "Qwen3-8B-8bit"
        assert fields["API key"] == "mlx-dspark"       # placeholder when none set


class TestKeyPlaceholder:
    def test_missing_key_becomes_a_placeholder_not_empty(self):
        """An empty api key field makes an OpenAI client send no Authorization header, which
        some servers reject outright — a placeholder is safer than blank."""
        generic = next(r for r in rows(key=None) if r["id"] == "openai-compatible")
        assert generic["setup"][0]["fields"]["API key"] == "mlx-dspark"
