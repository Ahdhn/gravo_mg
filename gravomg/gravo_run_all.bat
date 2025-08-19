@echo off
setlocal enabledelayedexpansion

REM === CONFIGURATION ===
set EXEC=gravomg_exe.exe
set CSV=timing_results.csv
set WRITE_HEADERS=True

REM === MESH LIST ===
set MESHES=dragon rocker-arm bumpy-cube nefertiti edgar-allen-poe beetle

for %%M in (%MESHES%) do (
    echo Running experiment on mesh: %%M

    set OBJ=data\%%M.obj
    set A=data\%%M_A.mtx
    set B=data\%%M_B.mtx

    %EXEC% !OBJ! !A! !B! %%M %CSV% %WRITE_HEADERS%

    set WRITE_HEADERS=false
)

pause
