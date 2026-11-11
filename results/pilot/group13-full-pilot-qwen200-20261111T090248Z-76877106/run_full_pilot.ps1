# Full Pilot, explicitly authorized DRAFT v0.1-dev; NOT final paper experiment.
# Actual input: data/processed/test_inputs.csv (all 200 instructions / 100 pairs).
# Logical jobs: Zero-shot=200, Few-shot=200, Checklist=200; retries max 2 per job.
# Run ID: group13-full-pilot-qwen200-20261111T090248Z-76877106
# Results: results/pilot/group13-full-pilot-qwen200-20261111T090248Z-76877106/
# Fixed model qwen-plus-2025-12-01; temperature=0; max_tokens=64; thinking=false.
# Effective runner: --split pilot --allow-draft-pilot --stop-on-anomaly
# --max-logical-requests 600 --prompt-version v0.1-dev --total-token-stop 300000
# Rerun this same script to resume. It skips all persisted final logical records.
# After budget-only stop: review usage, then use -TokenBudget 400000 -ExtendTokenBudget.
# Permission/failure/malformed-usage stops remain latched; budget extension cannot clear them.
[CmdletBinding()]
param(
    [ValidateRange(1, 2147483647)][int]$TokenBudget,
    [switch]$ExtendTokenBudget,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$pilotManifest = Join-Path $PSScriptRoot 'pilot_manifest.json'
Push-Location -LiteralPath $repoRoot
try {
    python -B -m scripts.full_pilot check --manifest $pilotManifest
    if ($LASTEXITCODE -ne 0) { throw 'Pilot preflight failed. No API request attempted.' }
    if ($CheckOnly) { return }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
        throw 'DASHSCOPE_API_KEY is absent in this PowerShell process; no API request attempted.'
    }
    $pilotArgs = @('-B', '-m', 'scripts.full_pilot', 'run', '--manifest', $pilotManifest,
        '--real-api', '--allow-draft-pilot', '--stop-on-anomaly')
    if ($PSBoundParameters.ContainsKey('TokenBudget')) {
        $pilotArgs += @('--total-token-stop', [string]$TokenBudget)
    }
    if ($ExtendTokenBudget) { $pilotArgs += '--extend-token-budget' }
    python @pilotArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'Pilot stopped or process failed. Inspect execution_summaries.jsonl and journals; keep this run ID.'
    }
}
finally { Pop-Location }
