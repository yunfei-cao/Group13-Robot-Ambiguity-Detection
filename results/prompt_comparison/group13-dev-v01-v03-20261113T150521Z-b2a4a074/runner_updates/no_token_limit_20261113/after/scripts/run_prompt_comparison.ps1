[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [switch]$CheckOnly,
    [switch]$NoTokenLimit
)
$ErrorActionPreference = 'Stop'
$planPath = (Resolve-Path -LiteralPath $Plan).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Push-Location -LiteralPath $repoRoot
try {
    $checkArgs = @('-B', '-m', 'scripts.prompt_v03_comparison', 'check', '--plan', $planPath)
    if ($NoTokenLimit) { $checkArgs += '--no-token-limit' }
    python @checkArgs
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; no API request sent.' }
    if ($CheckOnly) { return }
    if ([string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) { throw 'DASHSCOPE_API_KEY is absent.' }
    $runArgs = @('-B', '-m', 'scripts.prompt_v03_comparison', 'run', '--plan', $planPath, '--real-api')
    if ($NoTokenLimit) { $runArgs += '--no-token-limit' }
    python @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Run stopped or contains invalid results; inspect this run and retain its ID.' }
}
finally { Pop-Location }
