[CmdletBinding()]
param([switch]$CheckOnly, [int]$TokenBudget, [switch]$ExtendTokenBudget)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$plan = Join-Path $PSScriptRoot 'candidate/plan.json'
Push-Location -LiteralPath $repoRoot
try {
    python -B -m scripts.prompt_v03_comparison check --plan $plan
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; no API request sent.' }
    if ($CheckOnly) { return }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) { throw 'DASHSCOPE_API_KEY is absent.' }
    $runArgs = @('-B', '-m', 'scripts.prompt_v03_comparison', 'run', '--plan', $plan, '--real-api')
    if ($PSBoundParameters.ContainsKey('TokenBudget')) { $runArgs += @('--token-budget', [string]$TokenBudget) }
    if ($ExtendTokenBudget) { $runArgs += '--extend-token-budget' }
    python @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Run stopped or contains invalid results; inspect this run and retain its ID.' }
}
finally { Pop-Location }
