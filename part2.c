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
	
	//===============Image Vector Padding======================	
	int pad_amount_X = 1, pad_amount_Y = 1;
    int padded_size_X = data_size_X + 2, padded_size_Y = data_size_Y + 2;
	int padded_size = padded_size_X * padded_size_Y;
	float* padded_in = calloc(padded_size, sizeof(float));

	int pad_index = 1+padded_size_X, in_index = 0, out_index = 0;
	int last_in_row;
	int last_in_padded = pad_index + padded_size - padded_size_X - padded_size_X - 2;
	for(; pad_index < last_in_padded; pad_index+=2){ 
		last_in_row = pad_index + data_size_X;
		for(; pad_index < last_in_row; pad_index++)
			padded_in[pad_index] = in[in_index++];
	}
   	//=================Kernel Flip=========================== 
	int kernel_size = KERNX * KERNY;
    float* flipped_kernel = malloc(kernel_size*sizeof(float));
    for(int i = 0; i<kernel_size; i++){
              flipped_kernel[i] = kernel[kernel_size-i-1];
    }	
	
	// Result registers for storing results of dot products
    __m128 kernel_row = _mm_setzero_ps();
	__m128 matrix_row = _mm_setzero_ps();
	__m128 res_row = _mm_setzero_ps();
	__m128 kernel_row1 = _mm_loadu_ps(flipped_kernel);
    __m128 kernel_row2 = _mm_loadu_ps(flipped_kernel+KERNX);
    __m128 kernel_row3 = _mm_loadu_ps(flipped_kernel+2*KERNX);	
	__m128 sum_res = _mm_setzero_ps();
	const int dp_mask = 0b01110111;
	
	//=================Indexing Variables=====================		
	float* kernel_location = flipped_kernel; //Kernel location
	float* sum = malloc(4 * sizeof(float)); //Pointer to sum
	float* in_location = padded_in + pad_amount_X - 1 + (pad_amount_Y -1 )* padded_size_X;
	int out_location = 0; 
	int x = 0;
	int y = 0;


	// Find a way to write in_location and out_location based on the 
	//=================Main Convolution Loop=================
    for(; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
    	for(x = 0; x < data_size_X-3; x+=4){ // the x coordinate of the output location we're focusing on
			in_location = padded_in + pad_amount_X + x - 1 + (pad_amount_Y + y - 1) * padded_size_X;
			kernel_location = flipped_kernel;	
			
			kernel_row  = _mm_load1_ps(kernel_location);
			matrix_row = _mm_loadu_ps(in_location);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = matrix_row;
		
			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);		
			matrix_row = _mm_loadu_ps(in_location+1);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);

			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);		
			matrix_row = _mm_loadu_ps(in_location+2);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);
						
			in_location += padded_size_X;
			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);
			matrix_row = _mm_loadu_ps(in_location);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);
		
			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);		
			matrix_row = _mm_loadu_ps(in_location+1);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);
			
			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);		
			matrix_row = _mm_loadu_ps(in_location+2);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);
		
			in_location += padded_size_X;
			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);
			matrix_row = _mm_loadu_ps(in_location);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);
		
			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);		
			matrix_row = _mm_loadu_ps(in_location+1);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);
			
			kernel_location++;
			kernel_row = _mm_load1_ps(kernel_location);		
			matrix_row = _mm_loadu_ps(in_location+2);
			matrix_row = _mm_mul_ps(kernel_row,matrix_row);
			sum_res = _mm_add_ps(sum_res,matrix_row);
		
			_mm_storeu_ps((out+out_location), sum_res);
			out_location += 4;	
		}	
   		for(; x < data_size_X; x++){
            in_location = padded_in + pad_amount_X + x - 1 + (pad_amount_Y + y - 1) * padded_size_X;
            matrix_row = _mm_loadu_ps(in_location);
            res_row = _mm_dp_ps(matrix_row, kernel_row1, dp_mask);
            sum_res = res_row;
                          
            in_location += padded_size_X;        
            matrix_row = _mm_loadu_ps(in_location);
            res_row = _mm_dp_ps(matrix_row, kernel_row2, dp_mask);
            sum_res = _mm_add_ps(res_row, sum_res);
                        
            in_location += padded_size_X;        
            matrix_row = _mm_loadu_ps(in_location);
            res_row = _mm_dp_ps(matrix_row, kernel_row3, dp_mask);
            sum_res = _mm_add_ps(res_row, sum_res);
                        
            _mm_storeu_ps(sum, sum_res);
            out[out_location++] = sum[0];
        }        
		
	}
	free(padded_in);
	return 1;
}
