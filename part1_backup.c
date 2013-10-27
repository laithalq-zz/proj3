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
	
	int pad_amount_X = kern_cent_X+1, pad_amount_Y = kern_cent_Y;
    int padded_size_X = data_size_X + pad_amount_X*2, padded_size_Y = data_size_Y + pad_amount_Y*2;
    float* padded_in = calloc(padded_size_X*padded_size_Y, sizeof(float));

	for(int y = pad_amount_Y; y < pad_amount_Y + data_size_Y; y++){
		for(int x = 0; x < data_size_X; x++)
			padded_in[y*padded_size_X + x + pad_amount_X] = in[(y-pad_amount_Y) * data_size_X + x];
	}
   	//=================Kernel Flip=========================== 
	int padded_kernel_size = (KERNX+1) * KERNY; //Add an empty column to expediate vectorization 
	float* padded_flipped_kernel = calloc(padded_kernel_size, sizeof(float));
	for(int row = 0; row < KERNY; row++){
		for(int col = 0; col < KERNX; col++)
			padded_flipped_kernel[row*(KERNX+1) + col] = kernel[row*KERNX + col];
	}
    
	

	//=================Vectorization of Padded Kernel===========================
	__m128 kernel_1 = _mm_loadu_ps(padded_flipped_kernel);
        __m128 kernel_2 = _mm_loadu_ps(padded_flipped_kernel+4);
        __m128 kernel_3 = _mm_loadu_ps(padded_flipped_kernel+8);     

	__m128 matrix_row1 = _mm_setzero_ps();
	__m128 matrix_row2 = _mm_setzero_ps();
	__m128 matrix_row3 = _mm_setzero_ps();
	
	__m128 result1 = _mm_setzero_ps();
	__m128 result2 = _mm_setzero_ps();
	__m128 result3 = _mm_setzero_ps();

	__m128 sum1 = _mm_setzero_ps();
	__m128 sum2 = _mm_setzero_ps();
/**
	matrix_row1 = _mm_loadu_ps(padded_in+0-1+pad_amount_X+padded_size_X*(pad_amount_Y + 0-1));
        matrix_row2 = _mm_loadu_ps(padded_in+0-1+pad_amount_X+padded_size_X*(pad_amount_Y + 0));
        matrix_row3 = _mm_loadu_ps(padded_in+0-1+pad_amount_X+padded_size_X*(pad_amount_Y + 0+1));

        result1 = _mm_mul_ps(kernel_1, matrix_row1);
        result2 = _mm_mul_ps(kernel_2, matrix_row2);
        result3 = _mm_mul_ps(kernel_3, matrix_row3);

        sum1 = _mm_add_ps(result1,result2);
        sum2 = _mm_add_ps(sum1,result1);

	sum1 = _mm_hadd_ps(sum2,sum2);
	sum2 = _mm_hadd_ps(sum1,sum1);
	
	float *total = malloc(4);
	_mm_storeu_ps(total, sum2);

	printf("Sum is %f \n", *total);
*/
	printf("Kernel is : \n");
	for(int i = 0; i < padded_kernel_size; i++){
		printf("%f ",*(padded_flipped_kernel+i));
		if((i+1)%4 == 0)
			printf("\n");
	}

	int xView = 0;
	int yView = 0;
	printf("\nIn is at (%d,%d) \n", xView,yView);
	for(int i = -1; i < 2; i++){
		for(int j = -1; j < 3; j++){
			printf("%f ",*(padded_in+xView+j+pad_amount_X+padded_size_X*(pad_amount_Y+i+yView)));
		}
		printf("\n");
	}
    //=================Main Convolution Loop=================
    for(int y = 0; y < data_size_Y; y++){ // the x coordinate of the output location we're focusing on
    	for(int x = 0; x < data_size_X; x++){ // the y coordinate of the output location we're focusing on
		matrix_row1 = _mm_loadu_ps(padded_in+x-1+pad_amount_X+padded_size_X*(pad_amount_Y + y-1));
        	matrix_row2 = _mm_loadu_ps(padded_in+x-1+pad_amount_X+padded_size_X*(pad_amount_Y + y));
	        matrix_row3 = _mm_loadu_ps(padded_in+x-1+pad_amount_X+padded_size_X*(pad_amount_Y + y+1));

	        result1 = _mm_mul_ps(kernel_1, matrix_row1);
       	 	result2 = _mm_mul_ps(kernel_2, matrix_row2);
        	result3 = _mm_mul_ps(kernel_3, matrix_row3);

        	sum1 = _mm_add_ps(result1,result2);
        	sum2 = _mm_add_ps(sum1,result1);

        	sum1 = _mm_hadd_ps(sum2,sum2);
        	sum2 = _mm_hadd_ps(sum1,sum1);

        	float *total = malloc(4);
       		_mm_storeu_ps(total, sum2);
		
		*(out+x+y) = *total;


/*		for(int i = 0; i < KERNX; i++){ // kernel unflipped x coordinate
    			for(int j = 0; j < KERNY; j++){ // kernel unflipped y coordinate
    				out[x+y*data_size_X] += kernel[i+j*KERNX] * padded_in[(x + pad_amount_X + i - kern_cent_X) + (y + pad_amount_Y + j - kern_cent_Y) * padded_size_X]; 
				}
    		}
*/		
    	}
   }
	
	return 1;
}
