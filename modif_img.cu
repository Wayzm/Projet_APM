#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "FreeImage.h"

#define WIDTH 1920 // I genuinely don't know why these values exist
#define HEIGTH 1024
#define BPP 24
#define ui32 unsigned int
#define nb_threads 32

using namespace std;

//commentaire
__device__ int get_id(void)
{
  int thread_per_block = blockDim.x * blockDim.y * blockDim.z;

  int blockid = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

  int threadid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;

  int id = blockid * thread_per_block + threadid;

  return id;
}

__global__ void saturation_r(ui32* d_img, size_t size)
{
    //On recupere l'ide du thread
    int id = get_id();

    //On passe la case +2 (rouge) a 255 pour saturer la couleur rouge
    if(id < size)
    {
        d_img[id*3+0] = 255;
    }
}

__global__ void saturation_g(ui32* d_img, size_t size)
{
    //On recupere l'id du thread
    int id = get_id();

    //On passe la case +2 (vert) a 255 pour saturer la couleur verte
    if(id < size)
    {
	d_img[id*3+1] = 255;
    }
}

__global__ void saturation_b(ui32* d_img, size_t size)
{
    //On recupere l'id du thread
    int id = get_id();

    //On passe la case +2 (bleu) a 255 pour saturer la couleur bleu
    if(id < size)
    {
        d_img[id*3+2] = 255;
    }
}
__global__ void diapositive(ui32* d_img, ui32 width){

    // Compute thread id
    const ui32 x = threadIdx.x + blockDim.x * blockIdx.x;
    const ui32 y = threadIdx.y + blockDim.y * blockIdx.y;
    const ui32 idx = y * 3 *(width) + x;
    // c = 255 - c
    d_img[idx] = 255 - d_img[idx];
}

__global__ void one_color(ui32* dr, ui32* dg, ui32* db, ui32 width, const int n){

    // Compute thread id
    const ui32 x = threadIdx.x + blockDim.x * blockIdx.x;
    const ui32 y = threadIdx.y + blockDim.y * blockIdx.y;
    const ui32 idx = y * (width) + x;
    // Every pixels is set to 0 except :
    if(n == 1){ // blue
        dr[idx] = 0;
        dg[idx] = 0;
    } else if (n == 2){ // red
        dg[idx] = 0;
        db[idx] = 0;
    } else if(n == 3){ // green
        dr[idx] = 0;
        db[idx] = 0;
    }
}

__global__ void flou(ui32* d_img, size_t size, size_t width)
{
   //On recupere l'id du thread
   int id = get_id();

   //On creer des valeurs qui vont nous permettre de recuperer les valeurs pour la modification
   ui32 img0, img1, img2;
   if(id < size)
   {
      img0 = d_img[id*3+0];
      img1 = d_img[id*3+1];
      img2 = d_img[id*3+2];

   //On utilise plusieurs if pour être sur de ne pas sortir du tableau alloue
      if(id+1 < size)
      {
         img0 += d_img[id*3+0+3];
         img1 += d_img[id*3+1+3];
         img2 += d_img[id*3+2+3];
      }
      if(id-1 < size && id-1 > 0)
      {
         img0 += d_img[id*3+0-3];
         img1 += d_img[id*3+1-3];
         img2 += d_img[id*3+2-3];
      }
      if(id+width < size)
      {
         img0 += d_img[(id+width)*3+0];
         img1 += d_img[(id+width)*3+1];
         img2 += d_img[(id+width)*3+2];
      }
      if(id-width < size && id-width > 0)
      {
         img0 += d_img[(id-width)*3+0];
         img1 += d_img[(id-width)*3+1];
         img2 += d_img[(id-width)*3+2];
      }
   }

   //On fait la moyenne des valeurs obtenue
   img0 /= 5;
   img1 /= 5;
   img2 /= 5;

   //On copie la valeur dans les cases du tableau pour modifier chaque composantes des pixels
   d_img[id*3+0] = img0;
   d_img[id*3+1] = img1;
   d_img[id*3+2] = img2;
}
__global__ void horizontal_sym(ui32* d_img, ui32* d_tmp, ui32 width, ui32 height){

    // Compute thread id
    const ui32 x = threadIdx.x + blockDim.x * blockIdx.x;
    const ui32 y = threadIdx.y + blockDim.y * blockIdx.y;
    const ui32 idx = y * (3 * width) + x;
    // Compute target destination
    const ui32 xT = x;
    const ui32 yT = height - y;
    const ui32 idxT = yT * 3 * width + xT;
    // Flipping the image
    d_tmp[idxT] = d_img[idx];
}

__global__ void grey_img(ui32* d_img, ui32 width, ui32 height){

    //Compute thread id
    int id = get_id();

    //On verifie que le thread fait bien son calcul sur une case du tableau
    if(id < width*height)
    {
       //On fait le calcul pour faire une nuance de gris
       float val  = d_img[id*3+0] * 0.299 + d_img[id*3+1] * 0.587 + d_img[id*3+2] * 0.114;
       //On copie les valeurs calcule dans les cases du tableau
       d_img[id*3+0] = val;
       d_img[id*3+1] = val;
       d_img[id*3+2] = val;
   }
}

/*** DO NOT FORGET TO GREY SCALE FIRST***/
__global__ void sobel(ui32* dr, ui32* dg, ui32* db, ui32 width, ui32 height){
    // Compute thread id
    const ui32 x = threadIdx.x + blockDim.x * blockIdx.x;
    const ui32 y = threadIdx.y + blockDim.y * blockIdx.y;
    const ui32 idx = y * width + x;

    /*** Top and bottom idx ***/
    int top = idx - width;
    int bottom = idx + width;
    /*** SOBEL ALGORITHM ***/
    /**  Sobel matrix convolution threads of each block, horizontal change
    ** | -1 0 1 |   | (0 0) (1 0) (2 0) |
    ** | -2 0 2 | x | (0 1) (1 1) (2 1) |
    ** | -1 0 1 |   | (0 2) (1 2) (2 2) |
    **
    ** sobel vertical change
    ** | 1 2 1 |   | (0 0) (1 0) (2 0) |
    ** | 0 0 0 | x | (0 1) (1 1) (2 1) |
    ** |-1 -2 -1 |   | (0 2) (1 2) (2 2) |
    **/
    __syncthreads();
    int value_h = 0;
    int value_v = 0;
    // idx of the corners the eimage
    // const int top_left = 0;
    // const int top_right = width - 1;
    // const int bottom_left = (height - 1) * width;
    // const int bottom_right = (height - 1) * width + width - 1;
    // left
    if(x > 0){
        value_h -= 2 * dr[idx - 1];
    }
    // right
    if(x < width - 1){
        value_h += 2 * dr[idx + 1];
    }
    // top
    if(y > 0){
        value_v += 2 * dr[top];
    }
    // bottom
    if(y < height - 1){
        value_v -= 2 * dr[bottom];
    }
    // bottom right
    if(x != width - 1 && y != height - 1){
        value_h += dr[bottom + 1];
        value_v -= dr[bottom + 1];
    }
    //top left
    if(x != 0 && y != 0){
        value_h -= dr[top - 1];
        value_v += dr[top - 1];
    }
    // top right
    if(y != 0 && x != width - 1){
        value_h += dr[top + 1];
        value_v += dr[top + 1];
    }
    // bottom left
    if(x != 0 && y != width - 1){
        value_h -= dr[bottom - 1];
        value_v -= dr[bottom - 1];
    }
    if(value_h < 0){
        value_h = 0;
    }
    if(value_v < 0){
        value_v = 0;
    }

    const ui32 result = sqrtf(value_h * value_h + value_v * value_v);
    __syncthreads();
    dr[idx] = result;
    dg[idx] = result;
    db[idx] = result;
}

int main(int argc, char** argv){

    FreeImage_Initialise();
    const char* PathName="img.jpg";
    const char* PathDest="new_img.png";

    cudaError_t err, status;
    ui32* d_img;
    ui32* d_tmp;
    ui32* dr;
    ui32* dg;
    ui32* db;

    // load and decode a regular file
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);
    if(fif == FIF_UNKNOWN){
        perror("The image either has no signature or the recognition is failing.\n");
        exit(1);
    }
    FIBITMAP* bitmap = FreeImage_Load(fif, PathName, 0);
    if(!bitmap){
        perror("Failed image memory allocation.\n");
        exit(1);
    }

    const ui32 width = FreeImage_GetWidth(bitmap);
    const ui32 height = FreeImage_GetHeight(bitmap);
    const ui32 pitch = FreeImage_GetPitch(bitmap);
    const ui32 IMG_SIZE = sizeof(ui32) * width * height;
    fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

    // Array of IMG
    ui32* img = (ui32*)malloc(3 * IMG_SIZE);
    if(img == NULL){
        perror("Memory allocation for img array failed.\n");
        exit(1);
    }

    ui32* hr = (ui32*)malloc(IMG_SIZE);
    if(hr == NULL){
        perror("Memory allocation for temporary array failed.\n");
        exit(1);
    }
    ui32* hg = (ui32*)malloc(IMG_SIZE);
    if(hg == NULL){
        perror("Memory allocation for temporary array failed.\n");
        exit(1);
    }
    ui32* hb = (ui32*)malloc(IMG_SIZE);
    if(hb == NULL){
        perror("Memory allocation for temporary array failed.\n");
        exit(1);
    }

    //On alloue la memoire sur le GPU pour pouvoir utiliser ces tableaux plus tard
    // RED, BLUE and GREEN pixels of IMG on device
    cudaMalloc((void**)&d_img, 3 * IMG_SIZE);
    cudaMalloc((void**)&d_tmp, 3 * IMG_SIZE);
    cudaMalloc((void**)&dr, IMG_SIZE);
    cudaMalloc((void**)&dg, IMG_SIZE);
    cudaMalloc((void**)&db, IMG_SIZE);

   //On transforme notre image en tableau de pixel pour pouvoir le modifier
   BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
    for (ui32 y = 0U; y < height; ++y){
      BYTE *pixel = (BYTE*)bits;
      for (ui32 x = 0U; x < width; ++x){
        int idx = ((y * width) + x) * 3;
        img[idx + 0] = pixel[FI_RGBA_RED];
        img[idx + 1] = pixel[FI_RGBA_GREEN];
        img[idx + 2] = pixel[FI_RGBA_BLUE];
        pixel += 3;
      }
      // next line
      bits += pitch;
    }

    //On copie notre tableau qui contient les pixels de l'image sur le GPU et on verifie qu'aucune erreur ne
    // se produise
    err = cudaMemcpy(d_img,img,3*IMG_SIZE,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        printf("INIT : probleme dans cudaMemcpy de img --> d_img.\n");

    for(int i = 1; i < argc; i++){

        if(!strcmp(argv[i],"saturation_r")){
            //initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(width/32+1, height/32+1);
            //lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            saturation_r<<<Num_Blocks,Threads_Per_Blocks>>>(d_img,width*height);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"SATURATION ROUGE : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(img,d_img,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SATURATION ROUGE : probleme dans cudaMemcpy de d_img --> img.\n");
        }

        if(!strcmp(argv[i],"saturation_g")){
            //initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(3 * width/Threads_Per_Blocks.x, height/Threads_Per_Blocks.y);

            //lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            saturation_g<<<Num_Blocks,Threads_Per_Blocks>>>(d_img,width*height);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"SATURATION VERTE : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(img,d_img,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SATURATION VERTE : probleme dans cudaMemcpy de d_img --> img.\n");
        }

        if(!strcmp(argv[i],"saturation_b")){
            //initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(3 * width/Threads_Per_Blocks.x, height/Threads_Per_Blocks.y);

            //lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            saturation_b<<<Num_Blocks,Threads_Per_Blocks>>>(d_img,width*height);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"SATURATION BLUE : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(img,d_img,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SATURATION BLUE : probleme dans cudaMemcpy de d_img --> img.\n");
        }

        if(!strcmp(argv[i],"grey_img")){
            // initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(3 * width/Threads_Per_Blocks.x, height/Threads_Per_Blocks.y);
            // lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            grey_img<<<Num_Blocks,Threads_Per_Blocks>>>(d_img, width, height);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"GREY : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(img,d_img,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("GREY : probleme dans cudaMemcpy de d_img --> img.\n");
        }

        if(!strcmp(argv[i],"flou")){
            // initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(width/Threads_Per_Blocks.x+1, height/Threads_Per_Blocks.y+1);
            // lancement du kernel 100 fois pour que le flou s'applique bien
            // et copie de la mémoire modifé dans le kernel sur l'hote
            for(int k = 0; k < 10; k++){
                flou<<<Num_Blocks,Threads_Per_Blocks>>>(d_img,width*height,width);
                status = cudaGetLastError();
                if (status != cudaSuccess) {
                    fprintf(stderr,"FLOU : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
                }
            }
            err = cudaMemcpy(img,d_img,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("FLOU : probleme dans cudaMemcpy de d_img --> img.\n");
        }

        if(!strcmp(argv[i],"sym")){
            //initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(3 * width/Threads_Per_Blocks.x, height/Threads_Per_Blocks.y);

            //lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            horizontal_sym<<<Num_Blocks,Threads_Per_Blocks>>>(d_img,d_tmp,width,height);
            if (status != cudaSuccess) {
                fprintf(stderr,"SYM : Error : failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(img,d_tmp,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SYM : probleme dans cudaMemcpy de d_img --> img.\n");
        }

        if(!strcmp(argv[i],"sobel")){
            // initialisation de la grille et du bloc niveau de gris
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(3 * width/Threads_Per_Blocks.x, height/Threads_Per_Blocks.y);
            // Grey scale
            grey_img<<<Num_Blocks,Threads_Per_Blocks>>>(d_img, width, height);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"SOBEL : Error: failed to launch Grey scale kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(img,d_img,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SOBEL : probleme dans cudaMemcpy de d_img --> img.\n");
            // Separation des pixels
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    hr[y * width + x] = img[idx + 0];
                    hg[y * width + x] = img[idx + 1];
                    hb[y * width + x] = img[idx + 2];
                }
            }
            // Copies des pixels vers ke device
            err = cudaMemcpy(dr, hr, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("SOBEL : probleme dans cudaMemcpy de hr --> dr.\n");
            err = cudaMemcpy(dg, hg, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("SOBEL : probleme dans cudaMemcpy de hg --> dg.\n");
            err = cudaMemcpy(db, hb, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("SOBEL : probleme dans cudaMemcpy de hb --> db.\n");

            // Block and grid dims for sobel
            dim3 Sobel_Threads_Per_Blocks(3, 3);
            dim3 Sobel_Num_Blocks(width/Sobel_Threads_Per_Blocks.x, height/Sobel_Threads_Per_Blocks.y);
            //lancement du kernel sobel
            sobel<<<Sobel_Num_Blocks,Sobel_Threads_Per_Blocks>>>(dr,dg, db, width, height);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"SOBEL : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(hr, dr, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SOBEL : probleme dans cudaMemcpy de dr --> hr.\n");
            err = cudaMemcpy(hg, dg, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SOBEL : probleme dans cudaMemcpy de dg --> hg.\n");
            err = cudaMemcpy(hb, db, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("SOBEL : probleme dans cudaMemcpy de db --> hb.\n");
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    img[idx + 0] = hr[y * width + x];
                    img[idx + 1] = hg[y * width + x];
                    img[idx + 2] = hb[y * width + x];
                }
            }
        }
        if(!strcmp(argv[i],"rouge")){
            // initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(width/Threads_Per_Blocks.x + 1, height/Threads_Per_Blocks.y + 1);
            // Separation des pixels
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    hr[y * width + x] = img[idx + 0];
                    hg[y * width + x] = img[idx + 1];
                    hb[y * width + x] = img[idx + 2];
                }
            }
            // Copies des pixels vers ke device
            err = cudaMemcpy(dr, hr, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("ROUGE : probleme dans cudaMemcpy de hr --> dr.\n");
            err = cudaMemcpy(dg, hg, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("ROUGE : probleme dans cudaMemcpy de hg --> dg.\n");
            err = cudaMemcpy(db, hb, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("ROUGE : probleme dans cudaMemcpy de hb --> db.\n");
            // lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            one_color<<<Num_Blocks,Threads_Per_Blocks>>>(dr, dg, db, width, 2);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"ROUGE : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(hr, dr, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("ROUGE : probleme dans cudaMemcpy de dr --> hr.\n");
            err = cudaMemcpy(hg, dg, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("ROUGE : probleme dans cudaMemcpy de dg --> hg.\n");
            err = cudaMemcpy(hb, db, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("ROUGE : probleme dans cudaMemcpy de db --> hb.\n");
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    img[idx + 0] = hr[y * width + x];
                    img[idx + 1] = hg[y * width + x];
                    img[idx + 2] = hb[y * width + x];
                }
            }
        }
        if(!strcmp(argv[i],"vert")){
            // initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(width/Threads_Per_Blocks.x + 1, height/Threads_Per_Blocks.y + 1);
            // Separation des pixels
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    hr[y * width + x] = img[idx + 0];
                    hg[y * width + x] = img[idx + 1];
                    hb[y * width + x] = img[idx + 2];
                }
            }
            // Copies des pixels vers ke device
            err = cudaMemcpy(dr, hr, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("VERT : probleme dans cudaMemcpy de hr --> dr.\n");
            err = cudaMemcpy(dg, hg, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("VERT : probleme dans cudaMemcpy de hg --> dg.\n");
            err = cudaMemcpy(db, hb, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("VERT : probleme dans cudaMemcpy de hb --> db.\n");
            // lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            one_color<<<Num_Blocks,Threads_Per_Blocks>>>(dr, dg, db, width, 3);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"VERT : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(hr, dr, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("VERT : probleme dans cudaMemcpy de dr --> hr.\n");
            err = cudaMemcpy(hg, dg, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("VERT : probleme dans cudaMemcpy de dg --> hg.\n");
            err = cudaMemcpy(hb, db, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("VERT : probleme dans cudaMemcpy de db --> hb.\n");
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    img[idx + 0] = hr[y * width + x];
                    img[idx + 1] = hg[y * width + x];
                    img[idx + 2] = hb[y * width + x];
                }
            }
        }
        if(!strcmp(argv[i],"blue")){
            // initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(width/Threads_Per_Blocks.x + 1, height/Threads_Per_Blocks.y + 1);
            // Separation des pixels
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    hr[y * width + x] = img[idx + 0];
                    hg[y * width + x] = img[idx + 1];
                    hb[y * width + x] = img[idx + 2];
                }
            }
            // Copies des pixels vers ke device
            err = cudaMemcpy(dr, hr, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("BLUE : probleme dans cudaMemcpy de hr --> dr.\n");
            err = cudaMemcpy(dg, hg, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("BLUE : probleme dans cudaMemcpy de hg --> dg.\n");
            err = cudaMemcpy(db, hb, IMG_SIZE, cudaMemcpyHostToDevice);
            if(err != cudaSuccess)
                printf("BLUE : probleme dans cudaMemcpy de hb --> db.\n");
            // lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            one_color<<<Num_Blocks,Threads_Per_Blocks>>>(dr, dg, db, width, 1);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"BLUE : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(hr, dr, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("BLUE : probleme dans cudaMemcpy de dr --> hr.\n");
            err = cudaMemcpy(hg, dg, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("BLUE : probleme dans cudaMemcpy de dg --> hg.\n");
            err = cudaMemcpy(hb, db, IMG_SIZE, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("BLUE : probleme dans cudaMemcpy de db --> hb.\n");
            for(ui32 y = 0U; y < height; ++y){
                for(ui32 x = 0U; x < 3 * width; ++x){
                    int idx = ((y * width) + x) * 3;
                    img[idx + 0] = hr[y * width + x];
                    img[idx + 1] = hg[y * width + x];
                    img[idx + 2] = hb[y * width + x];
                }
            }
        }
        if(!strcmp(argv[i],"diapositive")){
            // initialisation de la grille et du bloc
            dim3 Threads_Per_Blocks(32, 32);
            dim3 Num_Blocks(3 * width/Threads_Per_Blocks.x + 1, height/Threads_Per_Blocks.y + 1);
            // lancement du kernel et copie de la mémoire modifé dans le kernel sur l'hote
            diapositive<<<Num_Blocks,Threads_Per_Blocks>>>(d_img, width);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr,"DIAPO : Error: failed to launch kernel (%s)\n",cudaGetErrorString(status));
            }
            err = cudaMemcpy(img,d_img,3*IMG_SIZE,cudaMemcpyDeviceToHost);
            if(err != cudaSuccess)
                printf("DIAPO : probleme dans cudaMemcpy de d_img --> img.\n");
        }

    if(!strcmp(argv[i],"pop_art"))
   {
       //Ici on va utiliser les streams afin de pouvoir faire chaque partie de l'image (separe en 4)
       // en parallèle et aucune partie de l'image n'aura a attendre la fin de l'execution des autres

       //Creation des streams
       cudaStream_t stream[5];
       for(int s = 1; s < 5; s++)
       {
          cudaStreamCreate(&stream[s]);
       }

       //On utilise la librairie pour rescale l'image pour en mettre 4 sur une seule image
       FIBITMAP *split = FreeImage_Rescale(bitmap,width/2,height/2,FILTER_BOX);

       //On récupere le pitch, la largeur et la hauteur de l'image modifie
       ui32 spitch = FreeImage_GetPitch(split);
       ui32 swidth = FreeImage_GetWidth(split);
       ui32 sheight = FreeImage_GetHeight(split);

       //malloc que l'on avait avant les streams, on ne les utilise plus car on a besoin de memoire punaise
       /*ui32* small = (ui32*)malloc(3*sizeof(ui32)*(width/2)*(height/2));
       ui32* bl = (ui32*)malloc(3*sizeof(ui32)*(width/2)*(height/2));
       ui32* br = (ui32*)malloc(3*sizeof(ui32)*(width/2)*(height/2));
       ui32* tl = (ui32*)malloc(3*sizeof(ui32)*(width/2)*(height/2));
       ui32* tr = (ui32*)malloc(3*sizeof(ui32)*(width/2)*(height/2));*/

       //On utilise cudaMallocHost pour faire de la mémoire punaise ce qui nous permettra de
       //faire des Memcpy asynchrone
       ui32* small;
       cudaMallocHost((void**)&small,3*sizeof(ui32)*(width/2)*(height/2));
       ui32* bl;
       cudaMallocHost((void**)&bl,3*sizeof(ui32)*(width/2)*(height/2));
       ui32* br;
       cudaMallocHost((void**)&br,3*sizeof(ui32)*(width/2)*(height/2));
       ui32* tl;
       cudaMallocHost((void**)&tl,3*sizeof(ui32)*(width/2)*(height/2));
       ui32* tr;
       cudaMallocHost((void**)&tr,3*sizeof(ui32)*(width/2)*(height/2));

       //On transforme l'image plus petite en tableau pour avoir acces a chaque pixels
       BYTE *bits = (BYTE*)FreeImage_GetBits(split);
       for (ui32 y = 0U; y < sheight; ++y){
          BYTE *pixel = (BYTE*)bits;
          for (ui32 x = 0U; x < swidth; ++x){
             int idx = ((y * (swidth)) + x) * 3;
             small[idx + 0] = pixel[FI_RGBA_RED];
             small[idx + 1] = pixel[FI_RGBA_GREEN];
             small[idx + 2] = pixel[FI_RGBA_BLUE];
             pixel += 3;
          }
          // next line
          bits += spitch;
       }

       ui32* dbl;
       ui32* dbr;
       ui32* dtl;
       ui32* dtr;


       //On malloc la memoire sur le GPU pour pouvoir faire nos calculs
       cudaMalloc((void**)&dbl, 3*sizeof(ui32)*swidth*sheight);
       cudaMalloc((void**)&dbr, 3*sizeof(ui32)*swidth*sheight);
       cudaMalloc((void**)&dtl, 3*sizeof(ui32)*swidth*sheight);
       cudaMalloc((void**)&dtr, 3*sizeof(ui32)*swidth*sheight);

       //On calcule la taille de la grille et du bloc
       dim3 Threads_Per_Blocks(32, 32);
       dim3 Num_Blocks(swidth/32+1, sheight/32+1);

       //Pour chaque partie de notre image on copie notre petite image dans un tableau specifique a
       //chaque stream, chaque stream fera ensuite l'operation demande sans attendre l'execution des autres
       //streams. Et enfin quand un stream a fini l'execution de son kernel il copie sa memoire dans un tableau
       //chaque tableau sera ensuite merge a un grand tableau pour reformer une image de la taille originale

       cudaMemcpyAsync(dbl,small,3*sizeof(ui32)*swidth*sheight,cudaMemcpyHostToDevice,stream[1]);
       saturation_r<<<Num_Blocks,Threads_Per_Blocks,0,stream[1]>>>(dbl,swidth*sheight);
       cudaMemcpyAsync(bl,dbl,3*sizeof(ui32)*swidth*sheight,cudaMemcpyDeviceToHost,stream[1]);

       cudaMemcpyAsync(dbr,small,3*sizeof(ui32)*swidth*sheight,cudaMemcpyHostToDevice,stream[2]);
       saturation_b<<<Num_Blocks,Threads_Per_Blocks,0,stream[2]>>>(dbr,swidth*sheight);
       cudaMemcpyAsync(br,dbr,3*sizeof(ui32)*swidth*sheight,cudaMemcpyDeviceToHost,stream[2]);

       cudaMemcpyAsync(dtl,small,3*sizeof(ui32)*swidth*sheight,cudaMemcpyHostToDevice,stream[3]);
       saturation_g<<<Num_Blocks,Threads_Per_Blocks,0,stream[3]>>>(dtl,swidth*sheight);
       cudaMemcpyAsync(tl,dtl,3*sizeof(ui32)*swidth*sheight,cudaMemcpyDeviceToHost,stream[3]);

       cudaMemcpyAsync(dtr,small,3*sizeof(ui32)*swidth*sheight,cudaMemcpyHostToDevice,stream[4]);
       grey_img<<<Num_Blocks,Threads_Per_Blocks,0,stream[4]>>>(dtr,swidth,sheight);
       cudaMemcpyAsync(tr,dtr,3*sizeof(ui32)*swidth*sheight,cudaMemcpyDeviceToHost,stream[4]);

       //On synchronize ici pour etre sur que tout les streams ont bien fait leur operations et
       //qu'ils ont bien copie leur tableaux sur le host

       cudaDeviceSynchronize();

       //Chaque double boucle for permet merge les tableau dans le tableau img pour refaire l'image a la fin

       for(int j = 0; j < width/2; j++)
       {
           for(int k = 0; k < height/2; k++)
           {
              img[(k*width+j)*3+0] = bl[(k*width/2+j)*3+0];
              img[(k*width+j)*3+1] = bl[(k*width/2+j)*3+1];
              img[(k*width+j)*3+2] = bl[(k*width/2+j)*3+2];
           }
       }

       for(int j = 0; j < width/2; j++)
       {
           for(int k = 0; k < height/2; k++)
           {
              img[(j+width/2+k*width)*3+0] = br[(k*width/2+j)*3+0];
              img[(j+width/2+k*width)*3+1] = br[(k*width/2+j)*3+1];
              img[(j+width/2+k*width)*3+2] = br[(k*width/2+j)*3+2];
           }
       }

       for(int j = 0; j < width/2; j++)
       {
           for(int k = 0; k < height/2; k++)
           {
              img[(j+(k+height/2)*width)*3+0] = tl[(k*width/2+j)*3+0];
              img[(j+(k+height/2)*width)*3+1] = tl[(k*width/2+j)*3+1];
              img[(j+(k+height/2)*width)*3+2] = tl[(k*width/2+j)*3+2];
           }
       }

       for(int j = 0; j < width/2; j++)
       {
           for(int k = 0; k < height/2; k++)
           {
              img[(j+width/2+(k+height/2)*width)*3+0] = tr[(k*width/2+j)*3+0];
              img[(j+width/2+(k+height/2)*width)*3+1] = tr[(k*width/2+j)*3+1];
              img[(j+width/2+(k+height/2)*width)*3+2] = tr[(k*width/2+j)*3+2];
           }
       }

    }

  }

   //On récupère la derniere erreur pour voir si le dernier appelle cuda qui a ete fait a fonctionne ou non
   err = cudaGetLastError();
   if(err != cudaSuccess)
        printf("probleme dans cudaMemcpy sortie: %s\n",cudaGetErrorString(err));

   //On retransforme notre tableau de bit en image
    bits = (BYTE*)FreeImage_GetBits(bitmap);
    for(int y =0; y<height; y++)
    {
        BYTE *pixel = (BYTE*)bits;
        for(int x = 0; x<width; ++x)
        {
            RGBQUAD newcolor;
            int idx = ((y * width) + x) * 3;
            newcolor.rgbRed = img[idx + 0];
            newcolor.rgbGreen = img[idx + 1];
            newcolor.rgbBlue = img[idx + 2];

            if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
                {fprintf(stderr, "(%d, %d) Fail...\n", x, y); }
        pixel+=3;
        }
        // next line
        bits += pitch;
    }

  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    cout << "Image successfully saved ! " << endl ;
  FreeImage_DeInitialise(); //Cleanup !

   // free(img);
   // free(h_img);
    return 0;
}
