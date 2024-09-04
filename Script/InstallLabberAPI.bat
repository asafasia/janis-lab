@echo off
set pythondir=%~1
if NOT "%pythondir%"=="" (
    setlocal
    SET "PATH=%pythondir%;%pythondir%\scripts\;%pythondir%\Library\bin;%PATH%"
    %pythondir%\scripts\pip.exe install . --no-input
) else (
pip install . --no-input
)