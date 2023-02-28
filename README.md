# Projet en Architecture et Programmation Parallèle


## Résumé

Ce projet a été ré&lisé par DIAS Nicolas, et YEUMO BARKWENDE Chutzpa William dans le cadre d'une unité d'enseignement pour l'apprentissage en programmation sur GPU.

## Instruction
### Build
```shell
make -j
```
### Utilisation
Assurer vous qu'un GPU qui supporte cuda est disponible sur votre plateforme
```shell
sbatch <exec>
```
## Fonctions

ui32 :
    unsigned int

d_img :
    L'array qui contient l'image entière sur le GPU

d_tmp :
    Un array alloué sur le GPU pour simplfier les opérations

width :
    Longueur de l'image

height :
    Largeur de l'image

hr / hg / hb :
    Arrays alloués sur le CPU contenant uniquement les pixels rouge / vert / bleu de l'image

dr / dg / db :
    Arrays alloués sur le GPU contenant uniquement les pixels rouge / vert / bleu de l'image


| Nom | Appel | Description |
| --- | ---------- | ----------- |
| __global__ void saturation_r(ui32* d_img, ui32 size) | saturation_r | Maximise la valeur des pixels rouge |
| __global__ saturation_b(ui32* d_img, ui32 size) | saturation_b | Maixmise la valeur des pixels bleu |
| __global__ void saturation_g(ui32* d_img, ui32 size) | saturation_g | Maximise la valeurs des pixels vert |
| __global__ grey_img(ui32* d_img, ui32 width, ui32 height) | grey_img | Transforme l'image en niveau de gris |
| __global__ flou(ui32* d_img, ui32 ???, ui32 width) | flou | Rend l'image flou |
| __global__ horizontal_sym(ui32* d_img, ui32* d_tmp, ui32 width, ui32 height) | sym | Symétrie horizontale de l'image |
| __global__ sobel(ui32* dr, ui32* dg, ui32* db, ui32 width, ui32 height) | sobel | Utilise le filtre sobel sur l'image |
| __global__ one_color(ui32* dr, ui32* dg, ui32* db, ui32 width, ui32 mode) | rouge / vert / bleu | Maximise les pixels d'une certaine couleurs et set les auters à 0 |