#include<stdio.h>
#include<nmmintrin.h>

int main(){
	

	float* matrix = calloc(12, sizeof(float));
	float* matrix2 = calloc(12, sizeof(float));
	matrix[0] = 0;
	matrix[1] = 1;
	matrix[2] = 2;
	matrix[3] = 3;
	matrix2[0] = 0;
	matrix2[1] = 1;
	matrix2[2] = 2;
	matrix2[3] = 3;
	

        __m128 kernel_1 = _mm_loadu_ps(matrix);
        __m128 kernel_2 = _mm_loadu_ps(matrix+4);
        __m128 kernel_3 = _mm_loadu_ps(matrix+8);

	__m128 kernel2_1 = _mm_loadu_ps(matrix2);
        __m128 kernel2_2 = _mm_loadu_ps(matrix2+4);
        __m128 kernel2_3 = _mm_loadu_ps(matrix2+8);

	const int mask = 127;
 	__m128 result1 = _mm_setzero_ps();
        __m128 result2 = _mm_setzero_ps();
        __m128 result3 = _mm_setzero_ps();
	

	result1 = _mm_mul_ps(kernel_1, kernel2_1);
        result2 = _mm_mul_ps(kernel_2, kernel2_2);
        result3 = _mm_mul_ps(kernel_3, kernel2_3);

	__m128 sum1 = _mm_setzero_ps();
	__m128 sum2 = _mm_add_ps(_mm_add_ps(result1,result2),result3);

	sum1 = _mm_hadd_ps(sum2,sum2);
	sum2 = _mm_hadd_ps(sum1,sum1);
	float* output = malloc(4);
	 _mm_storeu_ps(output,sum2);


	printf("Result: %f  %f  %f  %f",*output, *(output+1), *(output+2), *(output+3));



	}
