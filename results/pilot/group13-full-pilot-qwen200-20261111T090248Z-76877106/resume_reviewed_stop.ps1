[CmdletBinding()]
param([switch]$CheckOnly, [switch]$ResumeAfterReviewedStop)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$manifest = Join-Path $PSScriptRoot 'pilot_manifest.json'
Push-Location -LiteralPath $repoRoot
try {
    python -B -m scripts.resume_full_pilot --manifest $manifest --check-only
    if ($LASTEXITCODE -ne 0) { throw 'Recovery preflight failed; no API request attempted.' }
    if ($CheckOnly) { return }
    if (-not $ResumeAfterReviewedStop) {
        throw 'Explicit -ResumeAfterReviewedStop is required; no API request attempted.'
    }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
        throw 'DASHSCOPE_API_KEY is absent; no API request attempted.'
    }
    python -B -m scripts.resume_full_pilot --manifest $manifest --resume-after-reviewed-stop
    if ($LASTEXITCODE -ne 0) {
        throw 'Recovery stopped. Inspect journals; do not reset the authorization boundary.'
    }
}
finally { Pop-Location }
