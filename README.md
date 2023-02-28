# Projet en Architecture et Programmation Parallèle


## Résumé

Ce projet a été ré&lisé par DIAS Nicolas, et YEUMO BARKWENDE Chutzpa William dans le cadre d'une unité d'enseignement pour l'apprentissage en programmation sur GPU.

## Utilisation

## Fonctions

ui32 :
 unsigned int

d_img :
    L'array qui contient l'image entière sur le GPU


| Nom | Paramètres | Description |
| --- | ---------- | ----------- |
| __global__ void saturation_r(ui32* d_img, ui32 size) |  | Maximise la valeurs des pixels rouge |