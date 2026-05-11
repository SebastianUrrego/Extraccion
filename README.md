# Taller 1 Segundo Corte — Visión por Computador
**Universidad Sergio Arboleda**

Análisis comparativo de detectores y descriptores de características con Feature Matching y Homografía usando OpenCV en C++.

---

## Estructura del proyecto

```
Taller1corte2/
├── Data/
│   ├── box.png              ← imagen del objeto (caja de galletas)
│   └── box_in_scene.png     ← imagen de la escena
├── main_cv_features.cpp     ← código principal
└── README.md
```

---

## Requisitos

- OpenCV 4.3+ con módulo `xfeatures2d` (SURF, BRIEF, FREAK)
- CMake o QtCreator
- C++17

---

## Compilar

```bash
g++ main_cv_features.cpp -o taller1 \
    $(pkg-config --cflags --libs opencv4) \
    -lopencv_xfeatures2d -std=c++17
```

O desde **QtCreator**: abrir el proyecto y presionar Build.

> Si las imágenes no cargan, verificar que `PATH_OBJETO` y `PATH_SCENE` al inicio del archivo apunten a las rutas correctas.

---

## Algoritmos implementados

| Tipo | Algoritmos |
|---|---|
| Detectores | SIFT, SURF, ORB, FAST, BRISK (thresh 15 y 30) |
| Descriptores | SIFT, SURF, ORB, BRIEF (16/32/64 bytes), FREAK, BRISK |
| Matchers | Brute Force L2, Brute Force Hamming, FLANN KD-Tree |

---

## 50 combinaciones

| Rango | Grupo |
|---|---|
| 0 – 6 | SIFT detector |
| 7 – 14 | SURF detector |
| 15 – 22 | ORB detector |
| 23 – 26 | FAST detector |
| 27 – 34 | BRISK thresh=30 |
| 35 – 39 | BRISK thresh=15 |
| 40 – 44 | BRIEF 16 bytes (128 bits) |
| 45 – 49 | BRIEF 64 bytes (512 bits) |

---

## Output en terminal

Por cada combinación se imprime una línea:

```
[ID] NOMBRE ... [ OK ]    kp=XXXX good=XXX inliers=XXX t=XX.XXms
[ID] NOMBRE ... [FAILED]  kp=XXXX good=XXX inliers=XXX
```

Al final: tabla completa + ranking global.

**Criterio de éxito:** una combinación es `OK` si obtiene **4 o más inliers** tras aplicar RANSAC con `findHomography`.

---

## Resultado obtenido

```
Mejor inliers : [38] BRISK15+SURF+FLANN  (86 inliers, 140 good matches)
Mejor matches : [38] BRISK15+SURF+FLANN  (140 matches)
Más rápido    : [24] FAST+BRIEF+BF       (6.26 ms)
Total OK      : 40 / 50
```

Al terminar las 50 combinaciones el programa abre automáticamente **4 ventanas** con la mejor combinación:

1. Keypoints detectados sobre el objeto
2. Todos los matches sin filtrar
3. Buenos matches tras ratio test de Lowe (0.75)
4. Homografía con bounding box verde sobre la escena

Las 4 imágenes se guardan como PNG en la carpeta del ejecutable.

---

Imagen de Keypoints objetivos
<img width="492" height="423" alt="image" src="https://github.com/user-attachments/assets/18c85b5a-83de-4fc7-bdb2-29a315142068" />

Mejor Match
<img width="1272" height="522" alt="image" src="https://github.com/user-attachments/assets/abb3d70c-6253-4905-90bf-9da6c03675f6" />

---

## Reglas de compatibilidad

- **FAST** solo acepta descriptores binarios (ORB, BRIEF, FREAK, BRISK) — no tiene espacio de escala para SIFT/SURF.
- **FLANN** solo acepta descriptores flotantes (SIFT, SURF) — los binarios requieren BruteForce + Hamming.
- **SIFT + ORB** excluido — genera intento de alocar ~67 GB en RAM.
- **BRIEF** falla con todos los detectores — no tiene invarianza a rotación ni escala.
