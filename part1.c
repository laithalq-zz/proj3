#include <emmintrin.h>
#include <stdio.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
	
    // the x coordinate of the kernel's center
	int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
	
	int pad_amount_X = kern_cent_X, pad_amount_Y = kern_cent_Y;
    int padded_size_X = data_size_X + pad_amount_X*2, padded_size_Y = data_size_Y + pad_amount_Y*2;
    float* padded_in = calloc(padded_size_X*padded_size_Y, sizeof(float));

	for(int y = pad_amount_Y; y < pad_amount_Y + data_size_Y; y++){
		for(int x = 0; x < data_size_X; x++)
			padded_in[y*padded_size_X + x + pad_amount_X] = in[(y-pad_amount_Y) * data_size_X + x];
	}
   	//=================Kernel Flip=========================== 
	int kernel_size = KERNX * KERNY;
    float* flipped_kernel = calloc(kernel_size, sizeof(float));
    for(int i = 0; i < kernel_size; i++){
    	flipped_kernel[i] = kernel[kernel_size-1-i];
    }
		
    //=================Main Convolution Loop=================
    for(int y = 0; y < data_size_Y; y++){ // the x coordinate of the output location we're focusing on
    	for(int x = 0; x < data_size_X; x++){ // the y coordinate of the output location we're focusing on
    		for(int i = 0; i < KERNX; i++){ // kernel unflipped x coordinate
    			for(int j = 0; j < KERNY; j++){ // kernel unflipped y coordinate
    				out[x+y*data_size_X] += kernel[i+j*KERNX] * padded_in[(x + pad_amount_X + i - kern_cent_X) + (y + pad_amount_Y + j - kern_cent_Y) * padded_size_X]; 
				}
    		}
    	}
   	}
	
	return 1;
}
