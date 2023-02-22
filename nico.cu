#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
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
    int id = get_id();

    if(id < size)
    {
        d_img[id*3+0] = 255;
    }
}

__global__ void saturation_g(ui32* d_img, size_t size)
{
    int id = get_id();

    if(id < size)
    {
	d_img[id*3+1] = 255;
    }
}

__global__ void saturation_b(ui32* d_img, size_t size)
{
    int id = get_id();

    if(id < size)
    {
        d_img[id*3+2] = 255;
    }
}

__global__ void flou(ui32* d_img, size_t size, size_t width)
{
   int id = get_id();
   ui32 img0 = d_img[id*3+0];
   ui32 img1 = d_img[id*3+1];
   ui32 img2 = d_img[id*3+2];

   if(id+1 <= size)
   {
      img0 += d_img[id*3+0+3];
      img1 += d_img[id*3+1+3];
      img2 += d_img[id*3+2+3];
   }
   if(id-1 <= size)
   {
      img0 += d_img[id*3+0-3];
      img1 += d_img[id*3+1-3];
      img2 += d_img[id*3+2-3];
   }
   if(id+width <= size)
   {
      img0 += d_img[(id+width)*3+0];
      img1 += d_img[(id+width)*3+1];
      img2 += d_img[(id+width)*3+2];
   }
   if(id-width <= size)
   {
      img0 += d_img[(id-width)*3+0];
      img1 += d_img[(id-width)*3+1];
      img2 += d_img[(id-width)*3+2];
   }

   img0 /= 5;
   img1 /= 5;
   img2 /= 5;

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

__global__ void grey_img(ui32* dr, ui32* dg, ui32* db, ui32 width, ui32 height){

    // Compute thread id
    const ui32 x = threadIdx.x + blockDim.x * blockIdx.x;
    const ui32 y = threadIdx.y + blockDim.y * blockIdx.y;
    const ui32 idx = y * (width) + x;
    // Compute greyed pixel
    float value = dr[idx] * 0.299 + dg[idx] * 0.587 + db[idx] * 0.114;
    // Copy back the greyed pixel
    dr[idx] = value;
    dg[idx] = value;
    db[idx] = value;
}

int main(int argc, char** argv){

    FreeImage_Initialise();
    const char* PathName="img.jpg";
    const char* PathDest="new_img.png";

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
    ui32* h_img = (ui32*)malloc(3 * IMG_SIZE);
    if(h_img == NULL){
        perror("Memory allocation for temporary array failed.\n");
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
    // RED, BLUE and GREEN pixels of IMG on device
    cudaMalloc((void**)&d_img, 3 * IMG_SIZE);
    cudaMalloc((void**)&d_tmp, 3 * IMG_SIZE);
    cudaMalloc((void**)&dr, IMG_SIZE);
    cudaMalloc((void**)&dg, IMG_SIZE);
    cudaMalloc((void**)&db, IMG_SIZE);

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

    for(ui32 y = 0U; y < height; ++y){
        for(ui32 x = 0U; x < 3 * width; ++x){
            int idx = ((y * width) + x) * 3;
            hr[y * width + x] = img[idx + 0];
            hg[y * width + x] = img[idx + 1];
            hb[y * width + x] = img[idx + 2];
        }
    }

    cudaError_t err = cudaMemcpy(d_img,img,3*IMG_SIZE,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        printf("probleme dans cudaMemcpy");
    
    cudaMemcpy(dr, hr, IMG_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, hg, IMG_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, IMG_SIZE, cudaMemcpyHostToDevice);

    dim3 Threads_Per_Blocks(32, 32);
    dim3 Num_Blocks(width/Threads_Per_Blocks.x+1, height/Threads_Per_Blocks.y+1);
    
   horizontal_sym<<<Num_Blocks,Threads_Per_Blocks>>>(d_img,d_tmp,width,height);
   
   cudaMemcpy(hr, dr, IMG_SIZE, cudaMemcpyDeviceToHost);
   cudaMemcpy(hg, dg, IMG_SIZE, cudaMemcpyDeviceToHost);
   cudaMemcpy(hb, db, IMG_SIZE, cudaMemcpyDeviceToHost);
   /*for(ui32 y = 0U; y < height; ++y){
        for(ui32 x = 0U; x < 3 * width; ++x){
            int idx = ((y * width) + x) * 3;
            img[idx + 0] = hr[y * width + x];
            img[idx + 1] = hg[y * width + x];
            img[idx + 2] = hb[y * width + x];
        }
    }*/

   err = cudaGetLastError();
   if(err != cudaSuccess)
        printf("probleme dans cudaMemcpy");

   err = cudaMemcpy(img,d_tmp,3*IMG_SIZE,cudaMemcpyDeviceToHost);
   if(err != cudaSuccess)
        printf("probleme dans cudaMemcpy");

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

    free(img);
    free(h_img);
    return 0;
}
