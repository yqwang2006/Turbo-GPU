#include <iostream>
#include <fstream>
#include<time.h>
#include <cuda.h>
#include <stdlib.h> 
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#include "CuTurbo.h"

//常量存储器中每16个数为一组，共4组

__constant__ __device__ BYTE dev_nextS[64];
__constant__ __device__ BYTE dev_lastS[64];
__constant__ __device__ BYTE dev_vPos[64];
__constant__ __device__ bool dev_nextO[64];
__constant__ __device__ int dev_pi[MAXMN];


__global__ void  gpu_branch_metric1 ( T *d_v, int m_N, T *d_L, T *dev_rm, T *d_rv);
__global__ void  gpu_branch_metric ( T *dev_v, int m_N, T *dev_L, T *dev_rm, T *dev_rv);

__global__ void  gpu_maxlogmap( T *d_ialpha, T *d_ibeta, T *rm, T *rv, T *d_a, T *d_v, T *L, T *L_all, T *d_stmp, int m_N);

__global__ void  gpu_interweave(T *d_L_e, int m_N, T *d_L_all, T *d_rm, T *d_L_a);

__global__ void  gpu_interweave1(T* d_rm2, T *rm1, int m_N);

__global__ void  gpu_deinterweave1(bool *d_result, int m_N, T *L_al);

__global__ void  gpu_deinterweave(T *d_L_a, int m_N, T *d_L_e, T *d_L_all, T *d_rm);

__device__ inline T gpu_max_val( T a, T b );

void cpu_check( T *rv1, T *rv2, T *rm1, bool *m_Pattern, int m_Period, int m_N, int m_Len, int batch_len);

__global__ void gpu_check( T *d_rm1, T *d_rv1, T *d_rv2, T *d_input, bool *d_mPattern, double rate, int m_Period, int m_N, int m_Len);


template <class Type>
void print_file(Type *a, int size, string name){

	ofstream ofs;

	ofs.open(name);

	for(int i = 0; i < size; i++){

		ofs << a[i] << endl;

	}
	ofs << endl;

	ofs.close();

}

extern "C" int turbo_decoder(bool *result, T *input_code, int frame_len, int m_K, int m_Len, int m_Period, int *pi, bool *m_Pattern, BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16]){
	float costtime = 0;
	int batch_len = 0;
	int k = frame_len / BLOCKNUM;
	if( frame_len - k * BLOCKNUM > 0 ) k++;
	for(int f = 0; f < k; f++){

		if( (f+1)*BLOCKNUM < frame_len ){
			batch_len = BLOCKNUM;
		}else{
			batch_len = frame_len-f*BLOCKNUM;
		}
		cout << "Begin decoding batch " << f+1 << "/" << k << "...." << endl;

		gpu_decode(&result[m_K*f*BLOCKNUM], &input_code[f*m_Len*BLOCKNUM],batch_len, m_K, m_Len, m_Period, pi, m_Pattern, nextS, lastS, vPos, nextO, costtime);

		cout << "End batch "<< f+1 << "/" << k << "...." << endl;

	}

	costtime = costtime / 1000;

	cout << "It takes " << costtime<< " s" << endl;

	cout << "吞吐率为：" << (double)(frame_len * m_K) / (costtime * 1024 * 1024) << "Mbps" << endl;

	return 0;
}

extern "C" int gpu_decode(bool *result, T *input_code, int batch_len, int m_K, int m_Len, int m_Period, int *pi, bool *m_Pattern, BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16], float &costtime){

	cudaEvent_t start,stop;
	float duration;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}

	int m_N = m_K / 2;

	BYTE tmp[64];


	T *dev_rm1 = NULL;
	T *dev_rm2 = NULL;
	T *dev_rv1 = NULL;
	T *dev_rv2 = NULL;
	bool *dev_result = NULL;

	T *dev_La = NULL;
	T *dev_Le = NULL;
	T *dev_Lall = NULL;
	T *dev_ialpha1 = NULL;
	T *dev_ialpha2 = NULL;
	T *dev_ibeta1 = NULL;
	T *dev_ibeta2 = NULL;
	T *dev_stmp = NULL;
	T *dev_a = NULL;
	T *dev_v = NULL;
	T *dev_input = NULL;
	bool *dev_Pattern = NULL;

	int trible_mk = 3 * m_K * batch_len;
	unsigned int trible_mk_size = trible_mk * sizeof(T);
	unsigned int alpha_beta_size = nstate * batch_len * PATCHNUM * sizeof(T);

	cudaMalloc((void **)&dev_Pattern, m_Period * sizeof(bool) );
	cudaMalloc((void **)&dev_result, m_K * batch_len * sizeof(bool) );
	cudaMalloc((void **)&dev_input, m_Len * batch_len * sizeof(T) );
	cudaMalloc((void **)&dev_rm1, m_K * batch_len * sizeof(T));	
	cudaMalloc((void **)&dev_rm2, m_K * batch_len * sizeof(T));	
	cudaMalloc ((void **)&dev_rv1, m_N * batch_len * sizeof(T));
	cudaMalloc ((void **)&dev_rv2, m_N * batch_len * sizeof(T));
	cudaMalloc( (void **)&dev_La, trible_mk_size );
	cudaMalloc( (void **)&dev_Le, trible_mk_size );
	cudaMalloc( (void **)&dev_Lall, trible_mk_size );
	cudaMalloc( (void **)&dev_ialpha1, alpha_beta_size );
	cudaMalloc( (void **)&dev_ialpha2, alpha_beta_size );
	cudaMalloc( (void **)&dev_ibeta1, alpha_beta_size );
	cudaMalloc( (void **)&dev_ibeta2, alpha_beta_size );
	cudaMalloc( (void **)&dev_stmp, batch_len*m_N*sizeof(T) );
	cudaMalloc( (void **)&dev_a, nstate * (m_N+1) * batch_len * sizeof(T) );
	cudaMalloc( (void **)&dev_v, 8 * (m_N) * batch_len * sizeof(T) );

	cudaMemset( dev_rv1, 0, m_N * batch_len * sizeof(T) );
	cudaMemset( dev_rv2, 0, m_N * batch_len * sizeof(T) );
	cudaMemset( dev_La, 0, trible_mk_size );
	cudaMemset( dev_Le, 0, trible_mk_size );
	cudaMemset( dev_Lall, 0, trible_mk_size );
	cudaMemset( dev_ialpha1, 0, alpha_beta_size );
	cudaMemset( dev_ialpha2, 0, alpha_beta_size );
	cudaMemset( dev_ibeta1, 0, alpha_beta_size );
	cudaMemset( dev_ibeta2, 0, alpha_beta_size );



	//开始计时
	cudaEventRecord(start,0);

	for(int i = 0;i < 4;i++){
		for(int j = 0;j<16;j++){
			tmp[i*16+j] = nextS[i][j];
		}
	}
	cudaMemcpyToSymbol(dev_nextS, tmp, 64*sizeof(BYTE));

	for(int i = 0;i < 4;i++){
		for(int j = 0;j<16;j++){
			tmp[i*16+j] = lastS[i][j];
		}
	}
	cudaMemcpyToSymbol(dev_lastS, tmp, 64*sizeof(BYTE));

	for(int i = 0;i < 4;i++){
		for(int j = 0;j<16;j++){
			tmp[i*16+j] = vPos[i][j];
		}
	}
	cudaMemcpyToSymbol(dev_vPos, tmp, 64*sizeof(BYTE));

	bool tmp1[64];
	for(int i = 0;i < 4;i++){
		for(int j = 0;j<16;j++){
			tmp1[i*16+j] = nextO[i][j];
		}
	}
	cudaMemcpyToSymbol(dev_nextO, tmp1, 64*sizeof(bool));

	cudaMemcpyToSymbol(dev_pi, pi, m_N * sizeof(int));

	cudaMemcpy(dev_input, input_code, m_Len * batch_len * sizeof(T),cudaMemcpyHostToDevice);

	cudaMemcpy(dev_Pattern, m_Pattern, m_Period * sizeof(bool),cudaMemcpyHostToDevice);

	dim3 block_dim(PATCHNUM, batch_len);

	dim3 block_dim1(batch_len, 8);

	dim3 block_dim2(batch_len, 64);

	dim3 thread_dim(THREADCHUNKSIZE, PATCHNUM);

	int sum = 0;

	for(int i = 0;i < m_Period; i++){

		if(m_Pattern[i] == 1)
			sum++;
	}

	double rate = (double)sum / m_Period;

	gpu_check<<<block_dim1, THREADS>>>(dev_rm1, dev_rv1, dev_rv2, dev_input, dev_Pattern,rate, m_Period, m_N, m_Len);

	gpu_interweave1 <<<block_dim1, THREADS >>> (dev_rm2, dev_rm1, m_N);	

	//cudaEventRecord(stop,0);
	//cudaEventSynchronize( start );
	//cudaEventSynchronize( stop );
	//cudaEventElapsedTime(&duration,start,stop);
	//cout << "memcpy and interweave1 costs " << duration << " ms" << endl;
	//costtime += duration;

	int off_num = (int)m_N / THREADS + 1;

	int CHUNKSIZE = 0;

	int offset = 0;

	for(int iter = 0; iter < maxiter; iter++){

		//cudaEventRecord(start,0);

		gpu_branch_metric<<<block_dim2, THREADS>>>(dev_v,m_N,dev_La,dev_rm1,dev_rv1);

		/*cudaEventRecord(stop,0);
		cudaEventSynchronize( start );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime(&duration,start,stop);
		cout << "branch11 costs " << duration << " ms" << endl;
		costtime += duration;
		cudaEventRecord(start,0);*/

		gpu_maxlogmap <<< batch_len, thread_dim >>> (dev_ialpha1, dev_ibeta1, dev_rm1, dev_rv1, dev_a,dev_v, dev_La, dev_Lall,dev_stmp, m_N);

		/*cudaEventRecord(stop,0);
		cudaEventSynchronize( start );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime(&duration,start,stop);
		cout << "ff and bp costs " << duration << " ms" << endl;
		costtime += duration;
		cudaEventRecord(start,0);*/

		gpu_interweave <<<block_dim1, THREADS>>> ( dev_Le, m_N, dev_Lall, dev_rm1, dev_La );

		/*cudaEventRecord(stop,0);
		cudaEventSynchronize( start );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime(&duration,start,stop);
		cout << "interweave costs " << duration << " ms" << endl;
		costtime += duration;
		cudaEventRecord(start,0);*/

		gpu_branch_metric<<<block_dim2, THREADS>>>(dev_v,m_N,dev_Le,dev_rm2,dev_rv2);

		/*cudaEventRecord(stop,0);
		cudaEventSynchronize( start );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime(&duration,start,stop);
		cout << "branch2 costs " << duration << " ms" << endl;
		costtime += duration;
		cudaEventRecord(start,0);*/

		gpu_maxlogmap <<< batch_len, thread_dim >>> (dev_ialpha2, dev_ibeta2, dev_rm2, dev_rv2, dev_a,dev_v, dev_Le, dev_Lall,dev_stmp, m_N);
		/*cudaEventRecord(stop,0);
		cudaEventSynchronize( start );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime(&duration,start,stop);
		cout << "max2 costs " << duration << " ms" << endl;
		costtime += duration;
		cudaEventRecord(start,0);*/

		gpu_deinterweave <<<block_dim1, THREADS>>> ( dev_La, m_N, dev_Le, dev_Lall, dev_rm2 );

		/*cudaEventRecord(stop,0);
		cudaEventSynchronize( start );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime(&duration,start,stop);
		cout << "deinterweave costs " << duration << " ms" << endl;
		costtime += duration;
		cudaEventRecord(start,0);	*/
	}

	gpu_deinterweave1<<<block_dim1, THREADS>>>(dev_result, m_N, dev_Lall);

	cudaMemcpy( result, dev_result, m_K * batch_len * sizeof(bool), cudaMemcpyDeviceToHost); 

	cudaEventRecord(stop,0);
	cudaEventSynchronize( start );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime(&duration,start,stop);
	//cout << "Compute and memcpy costs " << duration << " ms" << endl;
	costtime += duration;

	cudaFree(dev_rm1);
	cudaFree(dev_rm2);
	cudaFree(dev_rv1);
	cudaFree(dev_rv2);
	cudaFree(dev_La);
	cudaFree(dev_Le);
	cudaFree(dev_Lall);
	cudaFree(dev_ialpha1);
	cudaFree(dev_ialpha2);
	cudaFree(dev_ibeta1);
	cudaFree(dev_ibeta2);
	cudaFree(dev_a);
	cudaFree(dev_v);
	cudaFree(dev_Pattern);
	cudaFree(dev_pi);

	return 0;
}
__global__ void  gpu_interweave1(T* d_rm2, T *d_rm1, int m_N){

	//TODO:
	int t,i,k, framebase_data, pi_val;

	bool pi_val_and_one;

	T rm11, rm12;

	framebase_data = 2 * m_N * blockIdx.x;

	t = blockIdx.y * blockDim.x + threadIdx.x;

	if( t < m_N ){

		pi_val = dev_pi[t];

		pi_val_and_one = (pi_val & 1);

		rm11 = d_rm1[framebase_data + 2*pi_val];

		rm12 = d_rm1[framebase_data + 2*pi_val + 1];

		d_rm2[framebase_data + 2*t] = rm12 * (pi_val_and_one) + rm11 * ( 1 - pi_val_and_one );

		d_rm2[framebase_data + 2*t + 1] = rm11 * (pi_val_and_one) + rm12 * ( 1 - pi_val_and_one );

	}


}
__global__ void gpu_check( T *d_rm1, T *d_rv1, T *d_rv2, T *d_input, bool *d_mPattern, double rate, int m_Period, int m_N, int m_Len)
{
	int m_K = m_N * 2;

	int frame_id = blockIdx.x;
	int t = blockIdx.y * blockDim.x + threadIdx.x;

	if( t < m_N )
	{
		d_rm1[frame_id * m_K + 2 * t] = d_input[frame_id * m_Len + 2 * t];
		d_rm1[frame_id * m_K + 2 * t + 1] = d_input[frame_id * m_Len + 2 * t + 1];
		int curr_loc = t * rate;
		for(int i = 0;i < t % m_Period; i++)
		{
			if( d_mPattern[i] == 1 ){
				curr_loc ++;
			}
		}
		curr_loc = frame_id * m_Len + m_K + curr_loc * 2;
		if (d_mPattern[t%m_Period]){
			d_rv1[frame_id * m_N + t] = d_input[curr_loc];
			d_rv2[frame_id * m_N + t] = d_input[curr_loc+1];

		}
	}
}
void cpu_check( T *rv1, T *rv2, T *rm1, bool *m_Pattern, int m_Period, int m_N, int m_Len, int batch_len){

	T *ptr;

	int m_K = m_N * 2;

	for(int i = 0; i < batch_len; i++){

		ptr = &rm1[i*m_Len] + m_K;

		for (int t = 0; t < m_N; t++){

			if (m_Pattern[t%m_Period]){

				rv1[i*m_N+t] = *ptr++;

				rv2[i*m_N+t] = *ptr++;

			}
		}	

	}


}



__device__ inline T gpu_max_val( T a, T b ){

	return a>b?a:b;

}

//<<< BLOCKNUM, m_N >>>
__global__ void  gpu_interweave(T *d_L_e, int m_N, T *d_L_all, T *d_rm, T *d_L_a){

	//TODO:
	int i, t, k, framebase_data, L_addr_base, pi_val;

	bool pi_and_one;

	T rm11, rm12, a_val, b_val, val;



	t = blockIdx.y * blockDim.x + threadIdx.x;

	if ( t < m_N ){

		framebase_data = 2 * m_N * blockIdx.x;

		L_addr_base = framebase_data * 3;

		pi_val = dev_pi[t];

		pi_and_one = (pi_val & 1);

		rm11 = d_rm[framebase_data + 2 * pi_val];

		rm12 = d_rm[framebase_data + 2 * pi_val+1];

		a_val = d_L_all[L_addr_base+3*pi_val] - 2 * rm12 - d_L_a[L_addr_base + 3*pi_val];

		b_val = d_L_all[L_addr_base+3*pi_val+1] - 2 * rm11 - d_L_a[L_addr_base + 3*pi_val+1];

		if( pi_and_one){
			d_L_e[L_addr_base+3*t+1] = a_val;
			d_L_e[L_addr_base+3*t] = b_val;
		}else{
			d_L_e[L_addr_base+3*t+1] = b_val;
			d_L_e[L_addr_base+3*t] = a_val;
		}

		d_L_e[L_addr_base+3*t+2] = d_L_all[L_addr_base+3*pi_val+2] - 2 * (rm11 + rm12) - d_L_a[L_addr_base + 3*pi_val+2];

	}


}

// <<< BLOCKNUM, M_N>>>
__global__ void  gpu_deinterweave(T *d_L_a, int m_N, T *d_L_e, T *d_L_all, T *d_rm){

	//TODO:


	int i, t, k, framebase_data, L_addr_base, pi_val;

	bool pi_and_one;

	T rm11, rm12, a_val, b_val;

	framebase_data = 2 * m_N * blockIdx.x;

	L_addr_base = framebase_data * 3;

	t = blockIdx.y * blockDim.x + threadIdx.x;

	if ( t < m_N ){

		pi_val = dev_pi[t];

		pi_and_one = (pi_val & 1);

		rm11 = d_rm[framebase_data + 2 * t];

		rm12 = d_rm[framebase_data + 2 * t+1];

		a_val = d_L_all[L_addr_base+3*t] - 2 * rm12 - d_L_e[L_addr_base + 3*t];

		b_val = d_L_all[L_addr_base+3*t+1] - 2 * rm11 - d_L_e[L_addr_base + 3*t+1];



		d_L_a[L_addr_base+3*pi_val+1] = a_val * (pi_and_one) + b_val * (1-pi_and_one);

		d_L_a[L_addr_base+3*pi_val] = a_val * (1-pi_and_one) + b_val * (pi_and_one);

		d_L_a[L_addr_base+3*pi_val+2] = d_L_all[L_addr_base+3*t+2] - 2 * (rm11+rm12) - d_L_e[L_addr_base + 3*t+2];

	}

}

// <<<BLOCKNUM, M_N>>>
__global__ void  gpu_deinterweave1(bool *d_result, int m_N, T *L_all){


	int i, j, k, t, l_addr_base, result_addr_base, maxi;

	T maxv;

	l_addr_base = 6*m_N*blockIdx.x;

	result_addr_base = 2*m_N*blockIdx.x;

	t = blockIdx.y * blockDim.x + threadIdx.x;

	maxv = 0;

	maxi = 0;

	if( t < m_N ){

		for( i = 1; i < 4; i++){

			if(L_all[l_addr_base+3*t+i-1] > maxv){

				maxv = L_all[l_addr_base+3*t+i-1];

				maxi = i;

			}

		}


		d_result[result_addr_base+2*dev_pi[t]+1] = (bool)((maxi >> 1)*(dev_pi[t]&1) + (maxi & 1)*(1-(dev_pi[t]&1)));

		d_result[result_addr_base+2*dev_pi[t]] = (bool)((maxi >> 1)*(1-(dev_pi[t]&1)) + (maxi & 1)*(dev_pi[t]&1));

	}

}
__global__ void  gpu_branch_metric ( T *dev_v, int m_N, T *dev_L, T *dev_rm, T *dev_rv){

	int i, j, k, t,loc1, loc2, tid;
	T tmp1,tmp2, tmp3, tmp4,tmp5,tmp6, tmpL;
	int m_K = 2 * m_N;

	tid = blockIdx.y * blockDim.x + threadIdx.x;
	
	if( tid < m_N*8 )
	{
		t = tid>>3;

		i = tid - t * 8;

		loc1 = 2 * t + blockIdx.x * m_N * 2;

		loc2 = blockIdx.x * m_N * 6 + 3*t;
		//loc2 = 3 * gridDim.x * m_K + blockIdx.x * m_K + t;

		tmp1 = dev_rm[loc1];

		tmp2 = dev_rm[loc1+1];

		tmp3 = 	dev_rv[blockIdx.x * m_N + t];	

		tmp4 = dev_L[loc2];

		tmp5 = dev_L[loc2+1];

		tmp6 = dev_L[loc2+2];

		if( i == 0 ){
			tmpL = -tmp1 - tmp2 - tmp3;
		}else if(i == 1 ){
			tmpL = -tmp1 - tmp2 + tmp3;
		}else if(i == 2 ){
			tmpL = tmp4 -tmp1 + tmp2 - tmp3;
		}else if(i == 3 ){
			tmpL = tmp4 -tmp1 + tmp2 + tmp3;
		}else if(i == 4 ){
			tmpL = tmp5 +tmp1 - tmp2 - tmp3;
		}else if(i == 5 ){
			tmpL = tmp5 +tmp1 - tmp2 + tmp3;
		}else if(i == 6 ){
			tmpL = tmp6 +tmp1 + tmp2 - tmp3;
		}else if(i == 7 ){
			tmpL = tmp6 + tmp1 + tmp2 + tmp3;
		}
		
		loc1 = t * 8 * gridDim.x + blockIdx.x * 8 + i;

		dev_v[loc1] = tmpL;

	}
	
}

__global__ void  gpu_branch_metric1 ( T *dev_v, int m_N, T *dev_L, T *dev_rm, T *dev_rv){


	int i, j, k, t,loc1, loc2;
	T tmp1,tmp2, tmp3, tmp4, tmpL;
	int m_K = 2 * m_N;

	t = blockIdx.y * blockDim.x + threadIdx.x;

	if( t < m_N )
	{
		loc1 = 2 * t + blockIdx.x * m_N * 2;

		loc2 = blockIdx.x * m_N * 6 + 3*t;
		//loc2 = 3 * gridDim.x * m_K + blockIdx.x * m_K + t;

		tmp1 = dev_rm[loc1];

		tmp2 = dev_rm[loc1+1];

		tmp3 = 	dev_rv[blockIdx.x * m_N + t];	

		

		loc1 = t * gridDim.x * 8 + blockIdx.x * 8;

		//loc1 = blockIdx.x * m_N + t;

		dev_v[loc1] = -tmp1 - tmp2 - tmp3;
		loc1 = loc1 + 1;
		dev_v[loc1] = -tmp1 - tmp2 + tmp3;
		loc1 = loc1 + 1;

		tmp4 = dev_L[loc2];
		dev_v[loc1] = tmp4 -tmp1 + tmp2 - tmp3;
		loc1 = loc1 + 1;
		dev_v[loc1] = tmp4 -tmp1 + tmp2 + tmp3;
		loc1 = loc1 + 1;

		tmp4 = dev_L[loc2+1];
		dev_v[loc1] = tmp4 +tmp1 - tmp2 - tmp3;
		loc1 = loc1 + 1;
		dev_v[loc1] = tmp4 +tmp1 - tmp2 + tmp3;
		loc1 = loc1 + 1;

		tmp4 = dev_L[loc2+2];
		dev_v[loc1] = tmp4 +tmp1 + tmp2 - tmp3;
		loc1 = loc1 + 1;
		dev_v[loc1] = tmp4 + tmp1 + tmp2 + tmp3;

	}
}
__global__ void  gpu_maxlogmap( T *d_ialpha, T *d_ibeta, T *d_rm, T *d_rv, T *d_a, T *d_v, T *d_L, T *d_Lall, T *d_stmp, int m_N){

	int i,t,k,i1,k1,loc1,loc2,loc3,loc4,rv_base,a_base,start_t,end_t,patch_size;

	T tmp1,tmp2,tmp3,tmp4;

	//__shared__ T s_v[MAXMN/PATCHNUM*8];
	__shared__ T s_btmp[nstate*PATCHNUM];
	__shared__ T reduction[nstate*2*PATCHNUM];
	__shared__ T reduction1[nstate*2*PATCHNUM];
	__shared__ T tmp_data[nstate*PATCHNUM];
	
	int patch_id = threadIdx.y;
	int frame_id = blockIdx.x;
	int tid_x = threadIdx.x;
	int batch_len = gridDim.x;
	int tid_in_block = tid_x + threadIdx.y * blockDim.x;
	// branch metric
	patch_size = m_N / PATCHNUM;

	start_t = patch_id * patch_size;

	end_t = (patch_id + 1) * patch_size;

	if( end_t > m_N )
		end_t = m_N;

	if(tid_x < nstate){

		if(patch_id == 0){
			tmp_data[patch_id * nstate + tid_x] = d_ialpha[frame_id * PATCHNUM * nstate + (PATCHNUM-1)*nstate+tid_x];
		}
		else{
			tmp_data[patch_id * nstate + tid_x] = d_ialpha[frame_id * PATCHNUM * nstate + (patch_id-1)*nstate+tid_x];
		}

		d_a[start_t * batch_len * nstate + frame_id * nstate + tid_x] = tmp_data[patch_id * nstate + tid_x];

	}

	a_base = start_t * batch_len * nstate + frame_id * nstate;

	rv_base = start_t * batch_len * 8 + frame_id * 8;

	//__syncthreads();

	for(t = start_t; t < end_t; t++){

		if( tid_x < 32 ){
			tmp1 = tmp_data[patch_id * nstate + dev_lastS[tid_x]] + d_v[rv_base + dev_vPos[tid_x]];
			tmp2 = tmp_data[patch_id * nstate + dev_lastS[tid_x+32]] + d_v[rv_base + dev_vPos[tid_x+32]];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);
		}
		//__syncthreads();
		if( tid_x < 16 ){

			tmp1 = reduction[patch_id * 32 + tid_x];
			tmp2 = reduction[patch_id * 32 + tid_x+16];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);
			tmp_data[patch_id * nstate + tid_x] = reduction[patch_id * 32 + tid_x];

		}
		//__syncthreads();
		if( tid_x < 8 ){
			tmp1 = reduction[patch_id * 32 + tid_x];
			tmp2 = reduction[patch_id * 32 + tid_x+8];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);
		}
		//__syncthreads();
		if( tid_x < 4 ){
			tmp1 = reduction[patch_id * 32 + tid_x];
			tmp2 = reduction[patch_id * 32 + tid_x+4];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);
		}
		//__syncthreads();
		if( tid_x < 2 ){
			tmp1 = reduction[patch_id * 32 + tid_x];
			tmp2 = reduction[patch_id * 32 + tid_x+2];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);
		}
		//__syncthreads();
		if( tid_x == 0 ){
			tmp1 = reduction[patch_id * 32];
			tmp2 = reduction[patch_id * 32+1];
			reduction[patch_id * 32] = gpu_max_val(tmp1,tmp2);
			d_stmp[frame_id*m_N+t] = reduction[patch_id * 32];

		}
		//__syncthreads();
		a_base += batch_len*nstate;
		rv_base += batch_len*8;

		if( tid_x < nstate ){

			tmp_data[patch_id * nstate + tid_x] -= reduction[patch_id * 32];

			d_a[a_base + tid_x] = tmp_data[patch_id * nstate + tid_x];

		}
		__syncthreads();
	}


	if(tid_x < nstate){

		d_ialpha[frame_id * PATCHNUM * nstate + patch_id * nstate + tid_x] = tmp_data[patch_id * nstate + tid_x];

		if(patch_id == PATCHNUM-1)
			s_btmp[patch_id * nstate + tid_x] = d_ibeta[frame_id * PATCHNUM * nstate + tid_x];
		else
			s_btmp[patch_id * nstate + tid_x] = d_ibeta[frame_id * PATCHNUM * nstate + (patch_id+1)*nstate + tid_x];

	}

	__syncthreads();
	////backward
	i = tid_x >> 2;
	k = tid_x - i * 4;
	i1 = (tid_x + 32) >> 2;
	k1 = tid_x + 32 - i1 * 4;
	loc1 = tid_x >> 4;
	loc2 = (tid_x+32)>>4; 
	loc3 = k*16+i;
	loc4 = k1*16+i1;

	a_base = (end_t-1) * batch_len * nstate + frame_id * nstate;

	rv_base = (end_t-1) * batch_len * 8 + frame_id * 8;

	for(t = end_t-1; t >= start_t; t--){


		if(tid_x < 32){

			tmp1 = d_a[a_base + dev_lastS[loc3]] + d_v[rv_base + dev_vPos[loc3]] + s_btmp[patch_id * nstate + i];
			tmp2 = d_a[a_base + dev_lastS[loc4]] + d_v[rv_base + dev_vPos[loc4]] + s_btmp[patch_id * nstate + i1];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);

			tmp1 = s_btmp[patch_id * nstate + dev_nextS[tid_x]] + d_v[rv_base+dev_vPos[loc1*16+dev_nextS[tid_x]]];
			tmp2 = s_btmp[patch_id * nstate + dev_nextS[tid_x+32]] + d_v[rv_base+dev_vPos[loc2*16+dev_nextS[tid_x+32]]];
			reduction1[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);

		}
		//__syncthreads();
		if( tid_x < 16 ){

			tmp1 = reduction[patch_id * 32 + tid_x];
			tmp2 = reduction[patch_id * 32 + tid_x+16];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);

			tmp1 = reduction1[patch_id * 32 + tid_x];
			tmp2 = reduction1[patch_id * 32 + tid_x+16];
			reduction1[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);

			s_btmp[patch_id * nstate + tid_x] = reduction1[patch_id * 32 + tid_x] - d_stmp[frame_id*m_N+t];

		}
		//__syncthreads();
		if( tid_x < 8 ){

			tmp1 = reduction[patch_id * 32 + tid_x];
			tmp2 = reduction[patch_id * 32 + tid_x+8];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);

		}
		//__syncthreads();
		if( tid_x < 4 ){

			tmp1 = reduction[patch_id * 32 + tid_x];
			tmp2 = reduction[patch_id * 32 + tid_x+4];
			reduction[patch_id * 32 + tid_x] = gpu_max_val(tmp1,tmp2);

		}
		//__syncthreads();
		if( tid_x < 3 ){

			d_Lall[frame_id*6*m_N+3*t+tid_x] = reduction[patch_id * 32 + tid_x+1]-reduction[patch_id * 32];
			//d_Lall[t*6*m_N * batch+3*t+tid_x] = reduction[patch_id * 32 + tid_x+1]-reduction[patch_id * 32];

		}
		a_base -= batch_len*nstate;
		rv_base -= batch_len*8;
		//__syncthreads();
	} 

	if( tid_x < nstate ){

		d_ibeta[frame_id * nstate * PATCHNUM + patch_id * nstate + tid_x] = s_btmp[patch_id * nstate + tid_x];

	}

}

