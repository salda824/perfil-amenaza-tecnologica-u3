@echo off
REM ============================================================================
REM  Proyecto Integrador U3 - ejecuta todo en Windows (cmd)
REM  Uso:   ejecutar_todo.bat "C:\ruta\a\ged261-csv"      (la ruta de UCDP es opcional)
REM ============================================================================
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONPATH=%CD%\src
set TF_CPP_MIN_LOG_LEVEL=3

echo [1/6] Entorno virtual (.venv)
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
call ".venv\Scripts\activate.bat" || goto :error

echo [2/6] Dependencias (la primera vez tarda varios minutos)
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || goto :error

if not "%~1"=="" (
    echo [3/6] Reduciendo UCDP GED a Colombia
    python herramientas\preparar_ucdp.py "%~1" || goto :error
) else (
    echo [3/6] Sin ruta de UCDP: la variable x3 queda pendiente
)

echo [4/6] Pruebas automaticas
python -m pytest -q tests || goto :error

echo [5/6] Cuaderno completo (unos 15-20 minutos)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Proyecto_U3_Integrador.ipynb || goto :error

echo [6/6] Proyeccion 2027 con el artefacto guardado
python -m perfil_amenaza.pipeline predecir --anio 2026 || goto :error

echo.
echo Listo. Resultados en salidas_u3\ y cuaderno ejecutado en Proyecto_U3_Integrador.ipynb
exit /b 0

:error
echo.
echo *** Fallo el paso anterior. Revise el mensaje de error de arriba. ***
exit /b 1
