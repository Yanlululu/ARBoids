param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [ValidateSet(0, 1)][int]$Setting = 1,
    [int]$Seed = 42,
    [ValidateSet('source', 'paper')][string]$TerminationRule = 'source',
    [string]$Distribution = 'ARBoids-22.04',
    [switch]$Headless,
    [switch]$CaptureFrames
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Checkpoint)) {
    $Checkpoint = Join-Path $repoPath $Checkpoint
}
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$linuxRepo = & wsl.exe -d $Distribution -u root --exec wslpath -a -u $repoPath
if ($LASTEXITCODE -ne 0) { throw 'Cannot access the ARBoids WSL environment.' }
$linuxCheckpoint = & wsl.exe -d $Distribution -u root --exec wslpath -a -u $checkpointPath
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the checkpoint in WSL.' }
$trialArgs = @('vrx/run_experiment.py', '--checkpoint', $linuxCheckpoint.Trim(),
               '--setting', [string]$Setting, '--seed', [string]$Seed,
               '--termination-rule', $TerminationRule)
if ($Headless) { $trialArgs += '--headless' }
if ($CaptureFrames) { $trialArgs += '--capture-frames' }
& wsl.exe -d $Distribution -u root --cd $linuxRepo.Trim() --exec bash -c '
set -e
source /opt/arboids-runtime/vrx_ws/activate.bash
exec python -X utf8 -u "$@"
' arboids-vrx @trialArgs
if ($LASTEXITCODE -ne 0) { throw "VRX experiment failed (exit $LASTEXITCODE). Inspect vrx/results/." }
