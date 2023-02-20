#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "FreeImage.h"

#define WIDTH 1920 // I genuinely don't know why these values exist
#define HEIGTH 1024
#define BPP 24
#define ui32 unsigned int

using namespace std;


int main(int argc, char** argv){

    FreeImage_Initialise();
    const char* PathName="img.jpg";
    const char* PathDest="new_img.png";

    ui32* dr_img, db_img, dg_img, hr_img, hb_img, hg_img;

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
    const ui32 IMG_SIZE = width * height;
    fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

    // Array of IMG
    ui32* img = (ui32*)malloc(sizeof(ui32) * 3 * IMG_SIZE);
    if(img == NULL){
        perror("Memory allocation for img array failed.\n");
        exit(1);
    }
    ui32* h_tmp = (ui32*)malloc(sizeof(ui32) * 3 * IMG_SIZE);
    if(h_tmp == NULL){
        perror("Memory allocation for temporary array failed.\n");
        exit(1);
    }
    // RED, BLUE and GREEN pixels of IMG on host
    cudaMallocHost((void**)&hr_img, IMG_SIZE);
    cudaMallocHost((void**)&hb_img, IMG_SIZE);
    cudaMallocHost((void**)&hg_img, IMG_SIZE);
    // RED, BLUE and GREEN pixels of IMG on device
    cudaMalloc((void**)&dr_img, IMG_SIZE);
    cudaMalloc((void**)&db_img, IMG_SIZE);
    cudaMalloc((void**)&dg_img, IMG_SIZE);

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

    free(img);
    return 0;
}