# Independent Round 2; DRAFT v0.1-dev, pilot only.
# 600 logical jobs (200 per strategy); retries can add paid HTTP attempts.
# Rerun this same script to resume ONLY Round 2. Token soft stop: 300000.
[CmdletBinding()]
param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$manifest = Join-Path $PSScriptRoot 'pilot_manifest.json'
Push-Location -LiteralPath $repoRoot
try {
    python -B -m scripts.pilot_round2 check --manifest $manifest
    if ($LASTEXITCODE -ne 0) { throw 'Round 2 preflight failed; no API request attempted.' }
    if ($CheckOnly) { return }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
        throw 'DASHSCOPE_API_KEY is absent; no API request attempted.'
    }
    python -B -m scripts.pilot_round2 run --manifest $manifest --real-api --allow-draft-pilot --stop-on-anomaly
    if ($LASTEXITCODE -ne 0) { throw 'Round 2 stopped or has invalid results; inspect Round 2 journals. Keep this run ID.' }
}
finally { Pop-Location }
