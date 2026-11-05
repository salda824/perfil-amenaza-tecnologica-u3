# Generador del cuaderno `Proyecto_U3_Integrador.ipynb`

El cuaderno de la U3 se generó con estos scripts: las celdas de código y los títulos están en `parte_a.py` a
`parte_d.py`, y las interpretaciones ("Lectura") en `lecturas.py` / `lecturas_fijas.py`, que se escribieron **después**
de ejecutar el cuaderno y citan sus cifras.

```bash
cd herramientas/cuaderno
python lecturas.py                                   # escribe lecturas.json
python construir.py ../../Proyecto_U3_Integrador.ipynb   # arma el cuaderno (sin ejecutar)
cd ../.. && jupyter nbconvert --to notebook --execute --inplace Proyecto_U3_Integrador.ipynb
```

Si solo cambia texto (no código), basta con reconstruir en otra ruta y copiar las celdas markdown sobre el cuaderno
ya ejecutado, para no repetir los ~15 minutos de cálculo. Si cambian los resultados, hay que revisar a mano las cifras
de `lecturas.py`: no se actualizan solas.
