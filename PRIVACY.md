# CrystalFlow privacy notice

CrystalFlow does not request credentials, perform analytics, serve advertising, track users, or
share data across Dify workspaces.

## Model processing

`CrystalFlow Adaptive` checks active exact routes before invoking the Agent model. On a miss, it
uses the model and tools configured on that Agent in the same way as a Function Calling strategy.
The selected model receives the Agent instruction, current query, configured history/context, and
tool schemas. An active route hit does not invoke the Agent model. The routed tool can have its own
data handling or model usage, which remains governed by that tool and its provider.

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

The Adaptive strategy stores normalized request hashes, app/instruction/tool-contract
fingerprints, approved tool identities, model-supplied JSON tool arguments, consistency state, and
aggregate hit/token-savings estimates. It does not separately store the original query, complete
conversation history, retrieved content, tool output, credentials, or current runtime/form
parameters. A tool argument can itself contain some or all of the user's query, so administrators
must select only tools whose replayable arguments are appropriate for workspace-local persistence.
Selecting the strategy and its **Safe direct-answer tools** is an administrator's opt-in to this
bounded route learning.

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

Retiring a compute crystal disables it but intentionally retains immutable versions for rollback
and audit. CrystalFlow has no bulk-purge tool, and uninstalling the plugin does not promise to erase
its persisted KV records. Permanent removal requires a Dify administrator to delete the plugin's
managed persistence/storage according to the deployment's retention procedures.
