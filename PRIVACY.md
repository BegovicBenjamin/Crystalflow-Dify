# CrystalFlow privacy notice

CrystalFlow does not request credentials, perform analytics, serve advertising, track users, or
share data across Dify workspaces.

## Model processing

`progressive_run` invokes the Dify workspace model selected in that node only when an active crystal
does not produce a hit or when the configured example threshold starts a builder call. The selected
model receives the task description and current structured input. A builder call also receives the
bounded examples retained for that task. Data is processed through Dify and the selected model
provider, so the workspace administrator is responsible for that provider's configuration,
retention terms, and privacy policy. Warm crystal hits do not invoke a model.

## Persistent data

The plugin stores crystal names, descriptions, deterministic JSON programs, test vectors, version
metadata, and aggregate execution counters in Dify's plugin key-value storage. Dify scopes that
storage to the workspace and plugin identity; records can persist across plugin upgrades or
reinstallation.

`execute_crystal` processes runtime inputs and outputs in memory and does not store them.
`progressive_run` also avoids retaining them when **Enable learning** is disabled. When a workflow
administrator explicitly enables learning, CrystalFlow retains a bounded set of canonical JSON
inputs and model outputs under the configured namespace and task key so it can detect conflicts and
generate a tested candidate. It does not capture unrelated conversation history, user identity, or
files.

Inputs and expected outputs supplied as manual or learned test vectors are stored with the version
record as audit evidence. Do not enable learning or include secrets, personal data, health data,
financial data, or other sensitive information unless your organization has an appropriate lawful
basis and Dify retention policy.

Retiring a crystal disables it but intentionally retains immutable versions for rollback and audit.
CrystalFlow 0.2.0 has no bulk-purge tool, and uninstalling the plugin does not promise to erase its
persisted KV records. Permanent removal requires a Dify administrator to delete the plugin's managed
persistence/storage according to the deployment's retention procedures.
