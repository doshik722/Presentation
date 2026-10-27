# Здесь меряем в PowerPoint настоящую высоту текста на слайдах для калибровки оценки text_overflow в сборщике.
# Запуск (Windows, установлен PowerPoint), из папки репозитория:
#   powershell -ExecutionPolicy Bypass -File scripts\text_fit\measure_pptx.ps1 -Folder runs\text_fit\<дата>\decks
# Результат — text_fit.csv рядом с папкой decks: строка на каждый абзац текста (сколько строк он занял
# и какая высота у текста всего блока) и на каждую фигуру без текста (картинки, графики) — их рамки.
# Файлы открываются только для чтения и без окна; уже открытые в PowerPoint презентации не трогаются.
param(
    [Parameter(Mandatory = $true)][string]$Folder,
    [string]$Out = ""
)
$ErrorActionPreference = "Stop"
$Folder = (Resolve-Path $Folder).Path
if (-not $Out) { $Out = Join-Path (Split-Path $Folder -Parent) "text_fit.csv" }

function Inch([double]$points) { [math]::Round($points / 72.0, 3) }

$app = New-Object -ComObject PowerPoint.Application
$rows = New-Object System.Collections.Generic.List[object]
$files = Get-ChildItem -Path $Folder -Filter *.pptx | Sort-Object Name
$n = 0
foreach ($file in $files) {
    $n += 1
    Write-Host "[$n/$($files.Count)] $($file.Name)"
    $pres = $app.Presentations.Open($file.FullName, -1, 0, 0)  # только чтение, без окна
    try {
        $slideHeight = Inch $pres.PageSetup.SlideHeight
        foreach ($slide in $pres.Slides) {
            foreach ($shape in $slide.Shapes) {
                $base = [ordered]@{
                    file = $file.Name; slide = $slide.SlideIndex; shape = $shape.Name; type = $shape.Type
                    left = Inch $shape.Left; top = Inch $shape.Top; width = Inch $shape.Width; height = Inch $shape.Height
                    slide_height = $slideHeight
                    margin_top = ""; margin_bottom = ""; margin_left = ""; margin_right = ""
                    bound_top = ""; bound_height = ""; paragraph = ""; font_size = ""; lines = ""; text = ""
                }
                $hasText = $false
                if ($shape.HasTextFrame -eq -1) { $hasText = $shape.TextFrame.HasText -eq -1 }
                if (-not $hasText) {
                    $rows.Add([pscustomobject]$base)
                    continue
                }
                $frame = $shape.TextFrame
                $range = $frame.TextRange
                $base.margin_top = Inch $frame.MarginTop
                $base.margin_bottom = Inch $frame.MarginBottom
                $base.margin_left = Inch $frame.MarginLeft
                $base.margin_right = Inch $frame.MarginRight
                $base.bound_top = Inch $range.BoundTop
                $base.bound_height = Inch $range.BoundHeight
                $count = $range.Paragraphs().Count
                for ($i = 1; $i -le $count; $i++) {
                    $p = $range.Paragraphs($i)
                    $row = [ordered]@{}
                    foreach ($key in $base.Keys) { $row[$key] = $base[$key] }
                    $row.paragraph = $i
                    $row.font_size = $p.Font.Size
                    try { $row.lines = $p.Lines().Count } catch { $row.lines = $p.Lines(1, 1000).Count }
                    $row.text = $p.Text.Trim()
                    $rows.Add([pscustomobject]$row)
                }
            }
        }
    }
    finally {
        $pres.Close()
    }
}
if ($app.Presentations.Count -eq 0) { $app.Quit() }
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
$rows | Export-Csv -Path $Out -NoTypeInformation -Encoding UTF8
Write-Host "Готово: $($rows.Count) строк -> $Out"
