#include <nmmintrin.h>
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
	
    int padded_size_X = data_size_X + 2, padded_size_Y = data_size_Y + 2;
	int padded_size = padded_size_X * padded_size_Y;
	float* padded_in = calloc(padded_size_X*padded_size_Y, sizeof(float));
	
	float flipped_kernel[9];
	int pad_index = 1+padded_size_X, in_index = 0;
	int last_in_row;
	int last_in_padded = pad_index + padded_size - padded_size_X - padded_size_X - 2; 
	for(; pad_index < last_in_padded; pad_index+=2){
		last_in_row = pad_index + data_size_X;
		for(; pad_index < last_in_row; pad_index++)
			padded_in[pad_index] = in[in_index++];
	}
   	//=================Kernel Flip=========================== 
    for(int i = 0,j=8; i < 9; i++,j--){
		flipped_kernel[i] = kernel[j];
    }
	
	#pragma omp parallel for schedule(dynamic, 8)
	for(int y = 1; y <= data_size_Y; y++){ 
		int index = y*padded_size_X + 1, out_index = y*data_size_X - data_size_X;
 		int last = index + data_size_X;
                for(; index < last; index++){
                                out[out_index] = flipped_kernel[0] * padded_in[index - 1 - padded_size_X];
                                out[out_index] += flipped_kernel[1] * padded_in[index - padded_size_X];
                                out[out_index] += flipped_kernel[2] * padded_in[index + 1 - padded_size_X];
                                out[out_index] += flipped_kernel[3] * padded_in[index - 1];
                                out[out_index] += flipped_kernel[4] * padded_in[index];
                                out[out_index] += flipped_kernel[5] * padded_in[index +1];
                                out[out_index] += flipped_kernel[6] * padded_in[index - 1 + padded_size_X];
                                out[out_index] += flipped_kernel[7] * padded_in[index + padded_size_X];
                                out[out_index] += flipped_kernel[8] * padded_in[index + 1 + padded_size_X];
                        out_index++;
    	}
	}

	free(padded_in);	
	return 1;
}
