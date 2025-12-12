$ErrorActionPreference = "Stop"
if (-not (Test-Path "test_artifacts")) { New-Item -ItemType Directory -Path "test_artifacts" | Out-Null }
pytest --junitxml test_artifacts/junit.xml
Write-Host "Coverage HTML: test_artifacts/coverage_html/index.html"
