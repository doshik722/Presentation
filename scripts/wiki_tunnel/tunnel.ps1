# Здесь держим туннель Википедии с моего компьютера: прокси proxy.py и обратный SSH-проброс на сервер.
<#
Туннель Википедии (решение 10.11.2026, по согласованию с администратором сервера).

Что делает:
  1. запускает proxy.py (127.0.0.1:Port; пропускает только ru/en.wikipedia.org:443);
  2. держит ssh -R Port:127.0.0.1:Port на сервер — на сервере открывается 127.0.0.1:Port, ведущий к прокси;
     при обрыве переподключается через 5 с;
  3. не даёт компьютеру уснуть, пока скрипт работает (настройки питания не меняет).

Нужен вход на сервер по SSH-ключу (без пароля): скрипт запускает ssh с BatchMode=yes, иначе переподключение
ночью встало бы на запросе пароля. Ключ — один раз, см. docs/server.md, раздел «Туннель Википедии».

Запуск (PowerShell, из любой папки):
  powershell -ExecutionPolicy Bypass -File scripts\wiki_tunnel\tunnel.ps1 -Server user@host [-SshPort 22] [-Port 18080]
На сервере: WIKI_PROXY=http://127.0.0.1:18080 (тетрадки grpo.ipynb, sft_eval.ipynb). Остановить — Ctrl+C.
#>
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [int]$SshPort = 22,
    [int]$Port = 18080,
    [string]$Python = "python",
    [string[]]$SshArgs = @()
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$logFile = Join-Path $here "tunnel.log"

function Log([string]$message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding UTF8
}

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "Нет ssh: установите компонент Windows «Клиент OpenSSH» (Параметры → Приложения → Дополнительные компоненты)"
}

Add-Type -Namespace WikiTunnel -Name Power -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
[WikiTunnel.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function Start-WikiProxy {
    $arguments = @("`"$(Join-Path $here 'proxy.py')`"", "--port", $Port, "--log", "`"$(Join-Path $here 'wiki_proxy.log')`"")
    return Start-Process -FilePath $Python -ArgumentList $arguments -PassThru -WindowStyle Minimized
}

$proxy = Start-WikiProxy
Start-Sleep -Seconds 2
if ($proxy.HasExited) { throw "proxy.py не запустился (код $($proxy.ExitCode)); проверьте, что порт $Port свободен" }
Log "прокси запущен (PID $($proxy.Id)), 127.0.0.1:$Port"

try {
    while ($true) {
        if ($proxy.HasExited) {
            Log "прокси остановился — перезапуск"
            $proxy = Start-WikiProxy
            Start-Sleep -Seconds 2
        }
        Log "подключаюсь к $Server (порт SSH $SshPort), проброс сервер:127.0.0.1:$Port -> сюда"
        $sshArguments = @("-N", "-p", $SshPort,
            "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-R", "${Port}:127.0.0.1:${Port}") + $SshArgs + @($Server)
        & ssh @sshArguments
        Log "туннель оборвался (код $LASTEXITCODE), переподключение через 5 с"
        Start-Sleep -Seconds 5
    }
}
finally {
    if ($proxy -and -not $proxy.HasExited) { Stop-Process -Id $proxy.Id -Force }
    [WikiTunnel.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null  # снова можно спать
    Log "туннель остановлен"
}
