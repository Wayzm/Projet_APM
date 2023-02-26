#include <iostream>
#include <string.h>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <stdlib.h>
#include "FreeImage.h"

#define WIDTH 3840
#define HEIGHT 2160
#define BPP 24
#define ui32 unsigned int

using namespace std;

//commentaire
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
    /**  Sobel matrix x threads of each block, horizontal change
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
    ui32* img = (ui32*)malloc(sizeof(ui32) * 3 * IMG_SIZE);
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

    // 1024 threads per blocs in 2D
    dim3 Threads_Per_Blocks(3, 3);
    // Blocks for inverted image
    // dim3 Num_Blocks(3 * width/Threads_Per_Blocks.x, height/Threads_Per_Blocks.y);
    // Blocks for gray scale and sobel
    dim3 Num_Blocks(width/Threads_Per_Blocks.x + 1, height/Threads_Per_Blocks.y + 1);

    // Copy to device
    // cudaMemcpy(d_img, img, 3 * IMG_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, hr, IMG_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, hg, IMG_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, IMG_SIZE, cudaMemcpyHostToDevice);

    // Grey scale
    grey_img<<<Num_Blocks, Threads_Per_Blocks>>>(dr, dg, db, width, height);
    // cudaMemcpy(hr, dr, IMG_SIZE, cudaMemcpyDeviceToHost);
    // cudaMemcpy(hg, dg, IMG_SIZE, cudaMemcpyDeviceToHost);
    // cudaMemcpy(hb, db, IMG_SIZE, cudaMemcpyDeviceToHost);
    // for(ui32 y = 0U; y < height; ++y){
    //     for(ui32 x = 0U; x < 3 * width; ++x){
    //         int idx = ((y * width) + x) * 3;
    //         img[idx + 0] = hr[y * width + x];
    //         img[idx + 1] = hg[y * width + x];
    //         img[idx + 2] = hb[y * width + x];
    //     }
    // }
    // Sobel
    sobel<<<Num_Blocks, Threads_Per_Blocks>>>(dr, dg, db, width, height);
    // Copy to Host
    // cudaMemcpy(img, d_tmp, 3 * IMG_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr, dr, IMG_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hg, dg, IMG_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, db, IMG_SIZE, cudaMemcpyDeviceToHost);
    for(ui32 y = 0U; y < height; ++y){
        for(ui32 x = 0U; x < 3 * width; ++x){
            int idx = ((y * width) + x) * 3;
            img[idx + 0] = hr[y * width + x];
            img[idx + 1] = hg[y * width + x];
            img[idx + 2] = hb[y * width + x];
        }
    }

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

    return 0;
}
