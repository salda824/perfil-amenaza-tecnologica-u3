> Texto extraído del cuaderno **Proyecto_U1_Exploracion.ipynb** (markdown y salidas de texto; sin imágenes). Código completo en `U1_exploracion_codigo.py`.
> Parte 1 de 1.

#### Celda de código 1: ── CARGAR DATOS ──────────────────────────────────────────────────  (fuente completa en U1_exploracion_codigo.py)
Salida de la celda:
```text
=== COLUMNAS DEL DATASET ===
['week', 'region', 'country', 'admin1', 'event_type', 'sub_event_type', 'events', 'fatalities', 'population_exposure', 'disorder_type', 'centroid_latitude', 'centroid_longitude']

Filas totales: 68,045

Primeras filas:
         week         region     country     admin1  \
0  2019-12-28     South Asia  Bangladesh      Dhaka   
1  2019-12-28     South Asia  Bangladesh      Dhaka   
2  2019-12-28     South Asia  Bangladesh      Dhaka   
3  2019-12-28     South Asia  Bangladesh      Dhaka   
4  2019-12-28     South Asia  Bangladesh   Rajshahi   
5  2019-12-28  South America    Colombia  Antioquia   
6  2019-12-28  South America    Colombia  Antioquia   
7  2019-12-28  South America    Colombia  Atlantico   
8  2019-12-28  South America    Colombia  Atlantico   
9  2019-12-28  South America    Colombia      Cauca   

                   event_type    sub_event_type  events  fatalities  \
0  Violence against civilians            Attack       1           1   
1                       Riots      Mob violence       2           0   
2                    Protests  Peaceful protest       4           0   
3      Strategic developments           Arrests       1           0   
4                       Riots      Mob violence       2           0   
5                     Battles       Armed clash       2           0   
6  Violence against civilians            Attack       1           1   
7  Violence against civilians            Attack       1           1   
8                       Riots      Mob violence       1           0   
9  Violence against civilians            Attack       2           2   

  population_exposure           disorder_type  centroid_latitude  \
0              66,107      Political violence             23.819   
1              76,169      Political violence             23.819   
2             147,621          Demonstrations             23.819   
3                   0  Strategic developments             23.819   
4              56,634      Political violence             24.554   
5              14,172      Political violence              7.157   
6               9,111      Political violence              7.157   
7             803,116      Political violence             10.668   
8             190,181      Political violence             10.668   
9              16,581      Political violence              2.142   

   centroid_longitude  
0              90.220  
1              90.220  
2              90.220  
3              90.2
[... salida truncada: 146 caracteres más]
```

#### Celda de código 2: ── 1. CARGAR Y FILTRAR COLOMBIA ──────────────────────────────────  (fuente completa en U1_exploracion_codigo.py)
Salida de la celda:
```text
=== DATASET COLOMBIA 2020-2026 ===
Total registros: 17,491
Departamentos: 33
Período: 2020-01-04 → 2026-08-01
Total eventos: 33,336
Total fatalidades: 13,732

=== SUBTIPOS DE EVENTO ===
sub_event_type
Peaceful protest                      10813
Attack                                 7946
Armed clash                            4799
Looting/property destruction           1551
Violent demonstration                  1274
Mob violence                           1252
Abduction/forced disappearance         1225
Other                                  1130
Disrupted weapons use                  1058
Remote explosive/landmine/IED           924
Grenade                                 349
Change to group/activity                290
Air/drone strike                        257
Protest with intervention               173
Arrests                                 131
Excessive force against protesters       60
Sexual violence                          48
Shelling/artillery/missile attack        44
Agreement                                 8
Suicide bomb                              3
Headquarters or base established          1
Name: events, dtype: int64

=== PERFILES DE AMENAZA POR DEPARTAMENTO ===
perfil_amenaza
D    17
C    16
Name: count, dtype: int64

Detalle por departamento:
perfil_base               total  pct_A  pct_B  pct_C  pct_D perfil_amenaza
admin1                                                                    
Amazonas                     71  0.014  0.000  0.817  0.169              C
Antioquia                  3745  0.037  0.006  0.625  0.332              C
Arauca                     1581  0.093  0.002  0.769  0.136              C
Atlantico                  1065  0.003  0.000  0.199  0.798              D
Bogota, D.C.               2482  0.002  0.000  0.045  0.953              D
Bolivar                    2059  0.014  0.007  0.360  0.619              D
Boyaca                      184  0.033  0.000  0.185  0.783              D
Caldas                      416  0.002  0.000  0.130  0.868              D
Caqueta                     403  0.074  0.002  0.744  0.179              C
Casanare                    134  0.052  0.007  0.679  0.261              C
Cauca                      4048  0.055  0.024  0.763  0.157              C
Cesar                       686  0.029  0.006  0.522  0.443              C
Choco                      1149  0.045  0.014  0.637  0.304              C
Cordoba                     725  0.011  0.001  0.498  0.490              D
Cundinamarca      
[... salida truncada: 2258 caracteres más]
```
Figuras generadas: 3 imagen(es) (imagen omitida en este texto).
