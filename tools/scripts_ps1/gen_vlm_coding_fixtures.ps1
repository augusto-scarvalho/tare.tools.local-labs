param(
    [string]$OutputDir = "runs/vlm/LAB-VLM-001-2026-08-22/fixtures"
)

Add-Type -AssemblyName System.Drawing
$resolved = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
New-Item -ItemType Directory -Force -Path $resolved | Out-Null

function New-Canvas([int]$Width = 1100, [int]$Height = 650, [string]$Color = "#F4F6F8") {
    $bitmap = New-Object System.Drawing.Bitmap($Width, $Height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $graphics.Clear([System.Drawing.ColorTranslator]::FromHtml($Color))
    return @($bitmap, $graphics)
}

function Save-Canvas($Canvas, [string]$Name) {
    $bitmap, $graphics = $Canvas
    $graphics.Dispose()
    $path = Join-Path $resolved $Name
    $bitmap.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
}

$mono = New-Object System.Drawing.Font("Consolas", 19)
$monoBold = New-Object System.Drawing.Font("Consolas", 19, [System.Drawing.FontStyle]::Bold)
$ui = New-Object System.Drawing.Font("Segoe UI", 20)
$uiBold = New-Object System.Drawing.Font("Segoe UI", 24, [System.Drawing.FontStyle]::Bold)
$small = New-Object System.Drawing.Font("Segoe UI", 16)
$white = [System.Drawing.Brushes]::White
$dark = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml("#182230"))
$muted = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml("#52606D"))
$red = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml("#D92D20"))
$green = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml("#16803C"))

# 1. Stack trace
$c = New-Canvas -Color "#111827"; $b, $g = $c
$g.FillRectangle((New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml("#1F2937"))), 0, 0, 1100, 58)
$g.DrawString("Payments API - exception", $uiBold, $white, 24, 10)
$lines = @(
    "2026-08-22 12:41:09 ERROR request_id=req-4471",
    "System.NullReferenceException: Object reference not set",
    "  at Payments.PaymentService.Charge(Order order)",
    "  in C:\src\Payments\PaymentService.cs:line 132",
    "  at Payments.CheckoutController.Submit(Int64 cartId)",
    "  in C:\src\Payments\CheckoutController.cs:line 58",
    "Context: order=4471 customer=acme retry=0"
)
$y = 92; foreach ($line in $lines) { $brush = if ($line -match "NullReference|line 132") { $red } else { $white }; $g.DrawString($line, $mono, $brush, 30, $y); $y += 62 }
Save-Canvas $c "stack_trace.png"

# 2. UI layout bug: clipped checkout button outside a card.
$c = New-Canvas; $b, $g = $c
$g.DrawString("Checkout", $uiBold, $dark, 55, 35)
$cardBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
$border = New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml("#CBD5E1"), 3)
$g.FillRectangle($cardBrush, 95, 120, 760, 420); $g.DrawRectangle($border, 95, 120, 760, 420)
$g.DrawString("Shipping address", $uiBold, $dark, 135, 155)
$g.DrawString("Street", $small, $muted, 135, 225); $g.DrawRectangle($border, 135, 258, 620, 55)
$g.DrawString("City", $small, $muted, 135, 340); $g.DrawRectangle($border, 135, 373, 620, 55)
$button = New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml("#2563EB"))
$g.FillRectangle($button, 705, 470, 280, 72)
$g.DrawString("Complete checkout", $ui, $white, 735, 487)
Save-Canvas $c "ui_bug.png"

# 3. Before/after visual diff.
$c = New-Canvas; $b, $g = $c
$g.DrawString("BEFORE", $uiBold, $dark, 190, 30); $g.DrawString("AFTER", $uiBold, $dark, 735, 30)
$g.DrawLine((New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml("#94A3B8"), 3)), 550, 15, 550, 630)
foreach ($x in @(55, 605)) { $g.FillRectangle($cardBrush, $x, 100, 440, 430); $g.DrawRectangle($border, $x, 100, 440, 430); $g.DrawString("Production", $uiBold, $dark, $x + 35, 135); $g.DrawString("api-service", $ui, $muted, $x + 35, 205) }
$g.FillEllipse($red, 375, 130, 88, 44); $g.DrawString("3 alerts", $small, $white, 383, 137)
$g.FillRectangle($green, 120, 395, 300, 72); $g.DrawString("Deploy", $uiBold, $white, 225, 410)
$g.FillRectangle($red, 670, 395, 300, 72); $g.DrawString("Delete", $uiBold, $white, 770, 410)
$g.DrawString("Compare the two panels", $small, $muted, 420, 575)
Save-Canvas $c "visual_diff.png"

# 4. Terminal failure.
$c = New-Canvas -Color "#0B1020"; $b, $g = $c
$g.DrawString("pytest", $uiBold, $white, 28, 18)
$lines = @(
    "tests/api/test_auth.py::test_login FAILED",
    "",
    "E   AssertionError: expected status 200, got 500",
    "E   KeyError: 'user_id'",
    "",
    "src/api/auth.py:87: KeyError",
    "1 failed, 47 passed in 3.28s"
)
$y = 90; foreach ($line in $lines) { $brush = if ($line -match "FAILED|500|KeyError") { $red } else { $white }; $g.DrawString($line, $monoBold, $brush, 35, $y); $y += 66 }
Save-Canvas $c "terminal_failure.png"

$mono.Dispose(); $monoBold.Dispose(); $ui.Dispose(); $uiBold.Dispose(); $small.Dispose()
$dark.Dispose(); $muted.Dispose(); $red.Dispose(); $green.Dispose(); $border.Dispose(); $cardBrush.Dispose(); $button.Dispose()
Write-Output $resolved
