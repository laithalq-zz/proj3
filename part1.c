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
	
	int kernel_size = KERNX * KERNY;
	float* flipped_kernel = calloc(kernel_size, sizeof(float));
	for(int i = 0; i < kernel_size; i++){
		flipped_kernel[i] = kernel[kernel_size-1-i];
	}
	
/**
	printf("Original kernel:\n");
	for(int i = 0; i < kernel_size; i++){
		printf("%f ", kernel[i]);
	}
	printf("\nFlipped kernel: \n");
	for(int i = 0; i < kernel_size; i++){
		printf("%f ", flipped_kernel[i]);
	}
*/
    // main convolution loop
	for(int y = 0; y < data_size_Y; y++){ // the x coordinate of the output location we're focusing on
		for(int x = 0; x < data_size_X; x++){ // the y coordinate of the output location we're focusing on
			for(int j = 0; j < KERNY; j++){ // kernel unflipped x coordinate
				for(int i = 0; i < KERNX; i++){ // kernel unflipped y coordinate
					// only do the operation if not out of bounds
					if(x+i-kern_cent_X>-1 && x+i-kern_cent_Y<data_size_X && y+j-kern_cent_Y>-1 && y+j-kern_cent_Y<data_size_Y){
						//Note that the kernel is flipped
						out[x+y*data_size_X] += 
								kernel[i+j*KERNX] * in[(x+i-kern_cent_X) + (y+j-kern_cent_Y)*data_size_X];
					
}
				}
			}
		}
	}
	return 1;
}
