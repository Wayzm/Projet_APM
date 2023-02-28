# Projet en Architecture et Programmation Parallèle


## Résumé

Ce projet a été réalisé par DIAS Nicolas, et YEUMO BARKWENDE Chutzpa William dans le cadre d'une unitée d'enseignement pour l'apprentissage en programmation sur GPU.

## Instruction
### Build
```shell
make -j
```
### Utilisation
Assurer vous qu'un GPU qui supporte cuda est disponible sur votre plateforme
```shell
./modif_img_cuda.exe <Appel(s)>
```

Voir les différents appels définies plus bas.
## Fonctions

ui32 :
    unsigned int

size :
    Taille de l'image en pixels

d_img :
    Le tableau qui contient l'image entière sur le GPU

d_tmp :
    Un tableau alloué sur le GPU pour simplfier les opérations

width :
    Longueur de l'image

height :
    Largeur de l'image

hr / hg / hb :
    tableaux alloués sur le CPU contenant uniquement les pixels rouge / vert / bleu de l'image

dr / dg / db :
    tableaux alloués sur le GPU contenant uniquement les pixels rouge / vert / bleu de l'image


| Nom | Appel | Description |
| --- | ---------- | ----------- |
| __global__ void saturation_r(ui32* d_img, ui32 size) | saturation_r | Maximise la valeur des pixels rouge |
| __global__ saturation_b(ui32* d_img, ui32 size) | saturation_b | Maixmise la valeur des pixels bleu |
| __global__ void saturation_g(ui32* d_img, ui32 size) | saturation_g | Maximise la valeurs des pixels vert |
| __global__ grey_img(ui32* d_img, ui32 width, ui32 height) | grey_img | Transforme l'image en niveau de gris |
| __global__ flou(ui32* d_img, ui32 size, ui32 width) | flou | Rend l'image flou |
| __global__ horizontal_sym(ui32* d_img, ui32* d_tmp, ui32 width, ui32 height) | sym | Symétrie horizontale de l'image |
| __global__ sobel(ui32* dr, ui32* dg, ui32* db, ui32 width, ui32 height) | sobel | Utilise le filtre sobel sur l'image |
| __global__ one_color(ui32* dr, ui32* dg, ui32* db, ui32 width, ui32 mode) | rouge / vert / blue | Garde la valeur du pixels d'une certaine couleurs et set les autres à 0 |
| __global__ diapositive(ui32* d_img, ui32 width) | diapositive | Applique le filtre diapositif sur l'image |
