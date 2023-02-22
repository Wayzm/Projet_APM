#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "FreeImage.h"

#define WIDTH 3840
#define HEIGHT 2160
#define BPP 24
#define ui32 unsigned int

using namespace std;

__global__ void horizontal_sym(ui32* d_img, ui32* d_tmp, ui32 width, ui32 height){

    // Compute thread id
    const ui32 x = threadIdx.x + blockDim.x * blockIdx.x;
    const ui32 y = threadIdx.y + blockDim.y * blockIdx.y;
    const ui32 idx = y * width + x;
    // Compute target destination
    const ui32 xT = x;
    const ui32 yT = blockDim.y * blockIdx.y - y%blockDim.y;
    const ui32 idxT = yT * width + xT;
    // Flipping the image
    d_tmp[idx] = 0;
}
int main(int argc, char** argv){

    FreeImage_Initialise();
    const char* PathName="img.jpg";
    const char* PathDest="new_img.png";

    ui32* d_img;
    ui32* d_tmp;

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
    ui32* h_img = (ui32*)malloc(3 * IMG_SIZE);
    if(h_img == NULL){
        perror("Memory allocation for temporary array failed.\n");
        exit(1);
    }

    // RED, BLUE and GREEN pixels of IMG on device
    cudaMalloc((void**)&d_img, 3 * IMG_SIZE);
    cudaMalloc((void**)&d_tmp, 3 * IMG_SIZE);

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

    // 1024 threads per blocs in 2D
    dim3 Threads_Per_Blocks(32, 32);
    dim3 Num_Blocks(width/Threads_Per_Blocks.x, height/Threads_Per_Blocks.y);

    // Copy to device
    cudaMemcpy(d_img, img, 3 * IMG_SIZE, cudaMemcpyHostToDevice);
    //Horizontal Symetry
    horizontal_sym<<<Num_Blocks, Threads_Per_Blocks>>>(d_img, d_tmp, width, height);
    // Copy to Host
    cudaMemcpy(img, d_tmp, 3 * IMG_SIZE, cudaMemcpyDeviceToHost);

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

    if(FreeImage_Save (FIF_PNG, bitmap , PathDest , 0))
        cout << "Image successfully saved ! " << endl ;
    FreeImage_DeInitialise(); //Cleanup !

    free(img);
    free(h_img);
    return 0;
}