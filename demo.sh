#!/bin/bash
# Aegis <2min showcase: validate -> scripted 3/3 -> random 0/3 -> batch -> report.
set -e
UV="C:/Users/ASUS/.local/bin/uv.exe"
export UV_PROJECT_ENVIRONMENT=".venv-win"
$UV run aegis validate
$UV run aegis eval --model scripted --episodes 3 --seed 42
$UV run aegis eval --model random --episodes 3 --seed 7
$UV run aegis eval-batch --models scripted,random --robots franka --episodes 1 --seed 7 --parallel 2
echo "=== latest report ==="
ls outputs/*/report.json | head -5
