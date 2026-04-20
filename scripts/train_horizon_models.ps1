Set-Location "D:/QinBo/Sea-Interface-Temperature-Prodictor"

$horizons = @(2, 4, 6, 12)

foreach ($horizon in $horizons) {
    $expName = "horizon_${horizon}m"
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Training horizon model: $horizon months" -ForegroundColor Cyan
    Write-Host "Experiment: $expName" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    D:/anaconda/python.exe src/train.py `
        --output_months $horizon `
        --exp_name $expName `
        --spatial_downsample 4
}
