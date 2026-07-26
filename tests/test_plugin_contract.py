from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

import yaml
from dify_plugin import AgentStrategy, DifyPluginEnv, Tool, ToolProvider
from dify_plugin.core.plugin_registration import PluginRegistration

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain a YAML object")
    return value


class PluginContractTests(unittest.TestCase):
    def test_official_sdk_loads_plugin_in_a_fresh_process(self) -> None:
        script = "\n".join(
            (
                "from dify_plugin import DifyPluginEnv",
                "from dify_plugin.core.plugin_registration import PluginRegistration",
                "registration = PluginRegistration(DifyPluginEnv(_env_file=None))",
                "assert not registration.tools_mapping",
                "assert 'crystalflow_agent' in registration.agent_strategies_mapping",
                "assert 'progressive_function_calling' in "
                "registration.agent_strategies_mapping['crystalflow_agent'][1]",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )

    def test_fresh_sdk_loader_builds_a_complete_parameter_model(self) -> None:
        script = "\n".join(
            (
                "import sys",
                "from dify_plugin import DifyPluginEnv",
                "from dify_plugin.core.plugin_registration import PluginRegistration",
                "assert 'strategies.progressive_function_calling' not in sys.modules",
                "registration = PluginRegistration(DifyPluginEnv(_env_file=None))",
                "strategy_class = registration.agent_strategies_mapping"
                "['crystalflow_agent'][1]['progressive_function_calling'][1]",
                "params_class = strategy_class._invoke_progressive.__globals__"
                "['ProgressiveFunctionCallingParams']",
                "assert params_class.__pydantic_complete__ is True",
                "tool = {",
                "    'identity': {",
                "        'author': 'Acme',",
                "        'name': 'get_sop',",
                "        'provider': 'acme/sop/sop',",
                "        'label': {'en_US': 'Get SOP'},",
                "    },",
                "    'parameters': [],",
                "    'description': {",
                "        'human': {'en_US': 'Retrieve an SOP.'},",
                "        'llm': 'Retrieve an SOP.',",
                "    },",
                "    'runtime_parameters': {},",
                "    'provider_type': 'plugin',",
                "}",
                "params = params_class.model_validate({",
                "    'query': 'What is in SOP-42?',",
                "    'instruction': 'Use the configured knowledge tools.',",
                "    'model': {",
                "        'provider': 'test-provider',",
                "        'model': 'test-model',",
                "        'model_type': 'llm',",
                "        'mode': 'chat',",
                "    },",
                "    'tools': [tool],",
                "    'crystallizable_tools': [tool],",
                "    'threshold': 2,",
                "})",
                "assert params.query == 'What is in SOP-42?'",
                "assert params.model.provider == 'test-provider'",
                "assert params.tools[0].identity.provider == 'acme/sop/sop'",
                "assert params.crystallizable_tools[0].identity.name == 'get_sop'",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )

    def test_official_sdk_loads_the_complete_plugin(self) -> None:
        registration = PluginRegistration(DifyPluginEnv())
        self.assertEqual(registration.configuration.name, "crystalflow")
        self.assertIn("crystalflow_agent", registration.agent_strategies_mapping)
        self.assertIn(
            "progressive_function_calling",
            registration.agent_strategies_mapping["crystalflow_agent"][1],
        )
        self.assertEqual(registration.tools_mapping, {})

    def test_manifest_declares_only_the_agent_strategy_plugin_type(self) -> None:
        manifest = load_yaml(ROOT / "manifest.yaml")
        self.assertEqual(manifest["type"], "plugin")
        self.assertEqual(manifest["author"], "begovicbenjamin")
        self.assertEqual(manifest["meta"]["runner"]["language"], "python")
        self.assertEqual(manifest["meta"]["runner"]["version"], "3.12")
        self.assertEqual(manifest["meta"]["minimum_dify_version"], "1.14.2")
        permission = manifest["resource"]["permission"]
        self.assertEqual(set(permission), {"model", "storage", "tool"})
        self.assertTrue(permission["model"]["enabled"])
        self.assertTrue(permission["model"]["llm"])
        self.assertTrue(permission["tool"]["enabled"])
        self.assertTrue(permission["storage"]["enabled"])

        self.assertEqual(set(manifest["plugins"]), {"agent_strategies"})
        agent_provider_paths = manifest["plugins"]["agent_strategies"]
        self.assertEqual(len(agent_provider_paths), 1)
        agent_provider = load_yaml(ROOT / agent_provider_paths[0])
        self.assertEqual(agent_provider["identity"]["author"], manifest["author"])
        self.assertEqual(len(agent_provider["strategies"]), 1)
        strategy = load_yaml(ROOT / agent_provider["strategies"][0])
        self.assertEqual(
            strategy["identity"]["name"],
            "progressive_function_calling",
        )
        self.assertIn("history-messages", strategy["features"])
        self.assertTrue((ROOT / strategy["extra"]["python"]["source"]).is_file())

    def test_sdk_can_import_every_entry_point(self) -> None:
        provider_module = importlib.import_module("provider.crystalflow")
        self.assertTrue(
            any(
                isinstance(value, type)
                and issubclass(value, ToolProvider)
                and value is not ToolProvider
                for value in vars(provider_module).values()
            )
        )

        for name in (
            "activate_crystal",
            "crystal_status",
            "crystallize",
            "execute_crystal",
            "progressive_run",
            "retire_crystal",
        ):
            with self.subTest(name=name):
                module = importlib.import_module(f"tools.{name}")
                self.assertTrue(
                    any(
                        isinstance(value, type) and issubclass(value, Tool) and value is not Tool
                        for value in vars(module).values()
                    )
                )

        strategy_module = importlib.import_module("strategies.progressive_function_calling")
        self.assertTrue(
            any(
                isinstance(value, type)
                and issubclass(value, AgentStrategy)
                and value is not AgentStrategy
                for value in vars(strategy_module).values()
            )
        )

    def test_marketplace_packaging_contract(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("dify-plugin==0.9.1", requirements)

        ignored = (ROOT / ".difyignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("__pycache__/", ignored)
        self.assertIn("*.py[cod]", ignored)
        self.assertIn("tools/", ignored)
        self.assertIn("provider/crystalflow.yaml", ignored)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("github.com/BegovicBenjamin/Crystalflow-Dify/issues", readme)


if __name__ == "__main__":
    unittest.main()
