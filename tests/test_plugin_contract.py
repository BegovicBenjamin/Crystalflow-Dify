from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from pathlib import Path

import yaml
from dify_plugin import DifyPluginEnv, Tool, ToolProvider
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
                "assert 'progressive_run' in registration.tools_mapping['crystalflow'][2]",
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
        self.assertEqual(
            set(registration.tools_mapping["crystalflow"][2]),
            {
                "activate_crystal",
                "crystal_status",
                "crystallize",
                "execute_crystal",
                "progressive_run",
                "retire_crystal",
            },
        )
        status_configuration = registration.tools_mapping["crystalflow"][2]["crystal_status"][0]
        include_program = next(
            parameter
            for parameter in status_configuration.parameters
            if parameter.name == "include_program"
        )
        self.assertIsNone(include_program.default)

    def test_manifest_provider_and_tools_are_wired(self) -> None:
        manifest = load_yaml(ROOT / "manifest.yaml")
        self.assertEqual(manifest["type"], "plugin")
        self.assertEqual(manifest["author"], "begovicbenjamin")
        self.assertEqual(manifest["meta"]["runner"]["language"], "python")
        self.assertEqual(manifest["meta"]["runner"]["version"], "3.12")
        self.assertEqual(manifest["meta"]["minimum_dify_version"], "1.14.2")
        permission = manifest["resource"]["permission"]
        self.assertEqual(set(permission), {"model", "storage"})
        self.assertTrue(permission["model"]["enabled"])
        self.assertTrue(permission["model"]["llm"])
        self.assertTrue(permission["storage"]["enabled"])

        provider_paths = manifest["plugins"]["tools"]
        self.assertEqual(len(provider_paths), 1)
        provider_path = ROOT / provider_paths[0]
        self.assertTrue(provider_path.is_file())
        provider = load_yaml(provider_path)
        self.assertEqual(provider["identity"]["author"], manifest["author"])
        self.assertTrue((ROOT / provider["extra"]["python"]["source"]).is_file())

        names: set[str] = set()
        for relative in provider["tools"]:
            tool_path = ROOT / relative
            self.assertTrue(tool_path.is_file(), relative)
            tool = load_yaml(tool_path)
            identity = tool["identity"]
            self.assertEqual(identity["author"], manifest["author"])
            self.assertNotIn(identity["name"], names)
            names.add(identity["name"])
            self.assertEqual(tool["output_schema"]["type"], "object")
            source = ROOT / tool["extra"]["python"]["source"]
            self.assertTrue(source.is_file(), source)
            for parameter in tool.get("parameters", []):
                self.assertIn(parameter["form"], {"llm", "form"})

        self.assertEqual(
            names,
            {
                "activate_crystal",
                "crystal_status",
                "crystallize",
                "execute_crystal",
                "progressive_run",
                "retire_crystal",
            },
        )

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

    def test_marketplace_packaging_contract(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("dify-plugin==0.9.1", requirements)

        ignored = (ROOT / ".difyignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("__pycache__/", ignored)
        self.assertIn("*.py[cod]", ignored)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("github.com/BegovicBenjamin/Crystalflow-Dify/issues", readme)


if __name__ == "__main__":
    unittest.main()
