# Generate the M0 VLM accept-test fixtures with KNOWN ground-truth text, so OCR is a clean
# pass/fail instead of a vibe check. Uses System.Drawing (always present on Windows).
Add-Type -AssemblyName System.Drawing

function New-Canvas([int]$w, [int]$h, [string]$bg) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $g.TextRenderingHint = 'ClearTypeGridFit'
    $g.Clear([System.Drawing.ColorTranslator]::FromHtml($bg))
    return @($bmp, $g)
}
function Draw-Text($g, [string]$s, [string]$font, [single]$size, [string]$color, [int]$x, [int]$y, [string]$style = 'Regular') {
    $f = New-Object System.Drawing.Font($font, $size, [System.Drawing.FontStyle]::$style)
    $b = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml($color))
    $g.DrawString($s, $f, $b, [single]$x, [single]$y)
    $f.Dispose(); $b.Dispose()
}
function Fill-Rect($g, [string]$color, [int]$x, [int]$y, [int]$w, [int]$h) {
    $b = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml($color))
    $g.FillRectangle($b, $x, $y, $w, $h); $b.Dispose()
}
function Draw-Rect($g, [string]$color, [int]$x, [int]$y, [int]$w, [int]$h) {
    $p = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml($color), 1)
    $g.DrawRectangle($p, $x, $y, $w, $h); $p.Dispose()
}

$dir = $PSScriptRoot

# --- 1) Error dialog (OCR target) --------------------------------------------------------
$c = New-Canvas 620 300 '#1f1f24'
$bmp, $g = $c
Fill-Rect $g '#2b2b33' 0 0 620 42
Draw-Text $g 'Application Error' 'Segoe UI' 12 '#f0f0f0' 16 12 'Bold'
Draw-Text $g 'X' 'Segoe UI' 12 '#c0c0c0' 590 12
Fill-Rect $g '#c0392b' 24 66 40 40
Draw-Text $g '!' 'Segoe UI' 22 '#ffffff' 39 70 'Bold'
Draw-Text $g 'Unhandled exception: NullReferenceException' 'Segoe UI' 11 '#f0f0f0' 80 62
Draw-Text $g 'at PaymentService.Charge(order=4471) line 132.' 'Consolas' 10 '#d8d8d8' 80 86
Draw-Text $g 'The transaction was not completed. Retry?' 'Segoe UI' 11 '#c8c8c8' 80 112
Fill-Rect $g '#3a7bd5' 360 240 110 34
Draw-Text $g 'Retry' 'Segoe UI' 11 '#ffffff' 392 248
Fill-Rect $g '#44444c' 484 240 110 34
Draw-Text $g 'Cancel' 'Segoe UI' 11 '#f0f0f0' 512 248
$bmp.Save((Join-Path $dir 'error_dialog.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

# --- 2) UI mockup (layout description target) --------------------------------------------
$c = New-Canvas 480 560 '#ffffff'
$bmp, $g = $c
Draw-Text $g 'Sign in to Acme' 'Segoe UI' 20 '#111111' 40 48 'Bold'
Draw-Text $g 'Welcome back. Please enter your details.' 'Segoe UI' 11 '#666666' 40 84
Draw-Text $g 'Email' 'Segoe UI' 11 '#333333' 40 140
Draw-Rect $g '#cccccc' 40 162 400 40
Draw-Text $g 'you@example.com' 'Segoe UI' 11 '#999999' 52 173
Draw-Text $g 'Password' 'Segoe UI' 11 '#333333' 40 224
Draw-Rect $g '#cccccc' 40 246 400 40
Draw-Text $g 'Forgot password?' 'Segoe UI' 10 '#3a7bd5' 320 300
Fill-Rect $g '#3a7bd5' 40 340 400 44
Draw-Text $g 'Continue' 'Segoe UI' 12 '#ffffff' 205 352 'Bold'
Draw-Text $g "Don't have an account?  Sign up" 'Segoe UI' 10 '#444444' 120 410
$bmp.Save((Join-Path $dir 'ui_mockup.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

Write-Output "wrote:"
Get-ChildItem $dir -Filter *.png | ForEach-Object { Write-Output ("  {0}  {1} bytes" -f $_.Name, $_.Length) }
