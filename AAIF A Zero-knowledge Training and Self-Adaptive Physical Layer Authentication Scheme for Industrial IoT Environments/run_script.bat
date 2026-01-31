@echo off
setlocal enabledelayedexpansion

REM =======================================
REM 基本配置
REM =======================================
set end_idx=count
set ranges=300_aap2_55

REM 创建日志目录
if not exist results mkdir results
if not exist results\log mkdir results\log

REM =======================================
REM 基础命令（固定部分）
REM =======================================
set "base_command=python main.py --epoch=50 --data_split=4_1 --use_autoencoder --model=adaiforest --channels=2 --two_channel=real_imag --seed=42 --count=16 --mode=train --normalization=minmax"

REM 提取模型名
for /f "tokens=3 delims== " %%a in ("%base_command%") do set "model=%%a"

REM 获取日期时间
for /f "tokens=1-3 delims=/- " %%a in ("%date%") do (
    set "y=%%a"
    set "m=%%b"
    set "d=%%c"
)
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set "h=%%a"
    set "min=%%b"
    set "s=%%c"
)
set h=0%h% & set h=%h:~-2%
set min=0%min% & set min=%min:~-2%
set s=0%s% & set s=%s:~-2%
set datetime=%y%%m%%d%_%h%%min%%s%

REM =======================================
REM 遍历每个 range
REM =======================================
for %%R in (%ranges%) do (
    set "range=%%R"

    echo ================================================
    echo Processing range: !range!
    echo ================================================

    set "logfile=results\log\%model%_!range!_%datetime%.log"

    (
        echo ==================================================
        echo Log file for range: !range!
        echo Model: %model%
        echo Base Command: %base_command%
        echo ==================================================
        echo.
    ) > "!logfile!"

    REM ===== 在这里调用子过程运行 idx 循环 =====
    call :run_idx_loop "%%R" "%end_idx%" "!logfile!"
)

echo.
echo All ranges completed successfully.
pause
exit /b

REM =======================================
REM 子过程：运行一个 range 的所有 idx
REM =======================================
:run_idx_loop
setlocal enabledelayedexpansion
set "range=%~1"
set "end_idx=%~2"
set "logfile=%~3"

set /a idx=1

:loop_idx
if !idx! gtr !end_idx! (
    echo Finished range: !range!
    echo Finished range: !range! >> "!logfile!"
    echo.
    endlocal
    exit /b
)

REM 拼接命令
set "full_command=%base_command% --range=!range! --idx=!idx!"

echo Running range=!range!, idx=!idx!
(
    echo Running range=!range!, idx=!idx!
    echo --------------------------------------------------
) >> "!logfile!"

REM 执行命令并写入日志
call !full_command! >> "!logfile!" 2>&1

if !errorlevel! neq 0 (
    echo Command failed with range=!range!, idx=!idx!
    echo Command failed with range=!range!, idx=!idx! >> "!logfile!"
    pause
    exit /b 1
) else (
    echo Command finished successfully: range=!range!, idx=!idx!
    echo Command finished successfully: range=!range!, idx=!idx! >> "!logfile!"
)

echo. >> "!logfile!"
set /a idx+=1
goto loop_idx
