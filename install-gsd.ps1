# Check if Node.js is installed
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "Node.js is not installed. Please install Node.js first." -ForegroundColor Red
    exit 1
}

# Check if npm is installed
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "npm is not installed. Please install npm first." -ForegroundColor Red
    exit 1
}

Write-Host "Installing get-shit-done-cc..." -ForegroundColor Cyan

# Install get-shit-done-cc globally using npx to ensure latest version and avoid path issues if possible, 
# but user specifically asked for an install script, so let's try global npm install or just npx usage.
# The search result recommended `npx get-shit-done-cc@latest`. 
# To "integrate" it effectively for a user to use commands like /gsd:help, 
# it's often run via npx. However, to make it "installed", global install is better.
npm install -g get-shit-done-cc

if ($LASTEXITCODE -eq 0) {
    Write-Host "get-shit-done-cc installed successfully!" -ForegroundColor Green
    Write-Host "You can now use commands like: npx gsd help" -ForegroundColor Cyan
} else {
    Write-Host "Installation failed." -ForegroundColor Red
}
