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
	int padded_kernel_size = KERNY * (KERNX + 1);
	float* padded_flipped_kernel = calloc(padded_kernel_size, sizeof(float));
    for(int row = 0; row < KERNY; row++){
    	for(int col = 0; col < KERNX; col++)
			padded_flipped_kernel[row*(KERNX+1) + col] = kernel[row*KERNX + col];
    }
    __m128 kernel_row1 = _mm_loadu_ps(padded_flipped_kernel);
    __m128 kernel_row2 = _mm_loadu_ps(padded_flipped_kernel+KERNX+1);
    __m128 kernel_row3 = _mm_loadu_ps(padded_flipped_kernel+2*(KERNX+1));

	__m128 matrix_row = _mm_setzero_ps();
	
	// Result registers for storing results of dot products
	__m128 res_row = _mm_setzero_ps();
	__m128 sum_res1 = _mm_setzero_ps();
	__m128 sum_res2 = _mm_setzero_ps();
	__m128 sum_res3 = _mm_setzero_ps();	
	const int dp_mask = 0b01110111;
	float* sum = malloc(4 * sizeof(float));
	float* in_location = padded_in + pad_amount_X - 1 + (pad_amount_Y -1 )* padded_size_X;
	int out_location = 0; 
	int x = 0;
	int y = 0;
	//=================Main Convolution Loop=================
    for(; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
    	for(; x < data_size_X-2; x+=3){ // the x coordinate of the output location we're focusing on
			in_location = padded_in + pad_amount_X + x - 1 + (pad_amount_Y + y - 1) * padded_size_X;
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row1, dp_mask);
			sum_res1 = res_row;
		
			in_location++;	
    		matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row1, dp_mask);
			sum_res2 = res_row;
			
			in_location++;	
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row1, dp_mask);
			sum_res3 = res_row;

			in_location -= 2;
			in_location+= padded_size_X; 

			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row2, dp_mask);
			sum_res1 = _mm_add_ps(res_row, sum_res1);

			in_location++;
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row2, dp_mask);
			sum_res2 = _mm_add_ps(res_row, sum_res2);

			in_location++;
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row2, dp_mask);
			sum_res3 = _mm_add_ps(res_row, sum_res3);
		
			in_location -= 2;
			in_location+= padded_size_X;			
	
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row3, dp_mask);
			sum_res1 = _mm_add_ps(res_row, sum_res1);
	
			in_location++;
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row3, dp_mask);
			sum_res2 = _mm_add_ps(res_row, sum_res2);
		
			in_location++;
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row3, dp_mask);
			sum_res3 = _mm_add_ps(res_row, sum_res3);

			_mm_storeu_ps(sum, sum_res1);
			out[out_location++] = sum[0];

			_mm_storeu_ps(sum, sum_res2);
			out[out_location++] = sum[0];
    		
			_mm_storeu_ps(sum, sum_res3);
			out[out_location++] = sum[0];	
    	}
		for(; x < data_size_X; x++){
			in_location = padded_in + pad_amount_X + x - 1 + (pad_amount_Y + y - 1) * padded_size_X;
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row1, dp_mask);
			sum_res1 = res_row;
			
			in_location += padded_size_X;	
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row2, dp_mask);
			sum_res1 = _mm_add_ps(res_row, sum_res1);
			
			in_location += padded_size_X;	
			matrix_row = _mm_loadu_ps(in_location);
			res_row = _mm_dp_ps(matrix_row, kernel_row3, dp_mask);
			sum_res1 = _mm_add_ps(res_row, sum_res1);
			
			_mm_storeu_ps(sum, sum_res1);
			out[out_location++] = sum[0];
		}	
		x = 0; 
   	}
	free(padded_in);
	return 1;
}
