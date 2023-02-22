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


int main(int argc, char** argv){

    FreeImage_Initialise();
    const char* PathName="img.jpg";
    const char* PathDest="new_img.png";

    ui32* d_img;

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
    ui32* img = (ui32*)malloc(sizeof(ui32) * 3 * IMG_SIZE);
    if(img == NULL){
        perror("Memory allocation for img array failed.\n");
        exit(1);
    }
    ui32* h_img = (ui32*)malloc(sizeof(ui32)*3 * IMG_SIZE);
    if(h_img == NULL){
        perror("Memory allocation for temporary array failed.\n");
        exit(1);
    }

    // RED, BLUE and GREEN pixels of IMG on device
    d_img = NULL;
    cudaMalloc((void**)&d_img, sizeof(ui32) * 3 * IMG_SIZE);
    if(!d_img)
    {
        printf("problème d'allocation memoire\n");
        exit(1);
    }

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

    cudaError_t err = cudaMemcpy(d_img,img,sizeof(ui32)*3*IMG_SIZE,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        printf("probleme dans cudaMemcpy");
    
    dim3 Threads_Per_Blocks(32, 32);
    dim3 Num_Blocks(width/Threads_Per_Blocks.x+1, height/Threads_Per_Blocks.y+1);
    
   saturation_g<<<Num_Blocks,Threads_Per_Blocks>>>(d_img,width*height); 
   err = cudaGetLastError();
   if(err != cudaSuccess)
        printf("probleme dans cudaMemcpy");

   err = cudaMemcpy(img,d_img,sizeof(ui32)*3*IMG_SIZE,cudaMemcpyDeviceToHost);
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
