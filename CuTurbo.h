#ifndef CPUTURBO_H
#define CPUTURBO_H


#define maxiter 5
#define nstate 16
#define BLOCKNUM 512
#define THREADCHUNKSIZE 32
#define THREADS 256
#define MAXMN 1752
#define PATCHNUM 8

typedef unsigned char BYTE;

typedef float T;

extern "C" int gpu_decode(bool *result, T *input_code, int batch_len, int m_K, int m_Len, int m_Period, int *pi, bool *m_Pattern, BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16], float &costtime);
extern "C" int turbo_decoder(bool *result, T *input_code, int frame_len, int m_K, int m_Len, int m_Period, int *pi, bool *m_Pattern, BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16]);

#endif
