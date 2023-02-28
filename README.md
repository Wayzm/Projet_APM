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


| Nom | Appel | Description |
| --- | ---------- | ----------- |
| __global__ void saturation_r(ui32* d_img, ui32 size) | saturation_r | Maximise la valeur des pixels rouge |
| __global__ saturation_b(ui32* d_img, ui32 size) | saturation_b | Maixmise la valeur des pixels bleu |
