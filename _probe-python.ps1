foreach ($n in 'py','python','python3','winget') {
  $c = Get-Command $n -ErrorAction SilentlyContinue
  '{0,-8} {1}' -f $n, $(if ($c) { $c.Source } else { '(ausente)' })
}
