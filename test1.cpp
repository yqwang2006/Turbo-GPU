#include <stdio.h>
#include <string>
#include <time.h>
#include "CuTurbo.h"

using namespace std;

void settrellis(BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16]);

void setinterleaver(int *pi, int P, int Q0, int Q1, int Q2, int Q3, int m_N);

void CTCC(int &WaveformNo, int &m_Period, bool *m_Pattern, T &m_Rate, int &m_K, int &m_N, int &m_rlen, int **pi, BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16]);


template <class Type>
void print_file(Type *a, int size, string name);



int main(){

	int waveformNo = 37;

	//cout << "Please input waveformNo (32-39) : " << endl;

	//cin >> waveformNo;

	int m_Period = 0, m_K = 0, m_N = 0, m_Len = 0;

	int *pi;

	float costtime = 0;

	bool m_Pattern[28];

	T m_Rate = 0;

	BYTE nextS[4][16];

	BYTE lastS[4][16];

	BYTE vPos[4][16];

	bool nextO[4][16];

	int frame_len  = 512;

	CTCC(waveformNo, m_Period, m_Pattern, m_Rate, m_K, m_N, m_Len, &pi, nextS, lastS, vPos, nextO);

	FILE *input_fp = fopen("llr_withouthead.dat","rb");
	FILE *output_fp = fopen("mc_result.dec","wb");

	fseek(input_fp, 0, SEEK_END);
	int nbytes = ftell(input_fp)/4;
	rewind(input_fp);

	frame_len = nbytes / m_Len;

	T *input = new T[m_Len*frame_len];

	bool *result = new bool[m_K*frame_len];

	printf( "Reading files ... ...\n") ;

	int reallen = fread(input, sizeof(T), nbytes, input_fp);


	printf( "Begin decoding \n") ;

	turbo_decoder(result, input,frame_len, m_K,  m_Len, m_Period, pi, m_Pattern, nextS, lastS, vPos, nextO);

	printf( "End Decoding\n") ;

	printf( "Writing files ... ...\n") ;

	fwrite(result, sizeof(bool), frame_len * m_K, output_fp);

	fclose(input_fp);
	fclose(output_fp);

	bool *groundTruth = new bool[m_K * frame_len];

	FILE *mc_fp = fopen("mc.dec","rb");

	BYTE *mc = new BYTE[m_K/8*frame_len];

	fread(mc,sizeof(BYTE),m_K/8*frame_len,mc_fp);

	fclose(mc_fp);
	for(int f = 0; f < frame_len; f++){
		for(int i = 0;i < m_K/8;i++){
			//ofs2 << groundTruth[i] << ' ';
			groundTruth[f*m_K+8*i] = (mc[f*m_K/8+i] & 128) >> 7; 

			groundTruth[f*m_K+8*i+1] = (mc[f*m_K/8+i] & 64) >> 6; 
			groundTruth[f*m_K+8*i+2] = (mc[f*m_K/8+i] & 32) >> 5; 
			groundTruth[f*m_K+8*i+3] = (mc[f*m_K/8+i] & 16) >> 4;  
			groundTruth[f*m_K+8*i+4] = (mc[f*m_K/8+i] & 8) >> 3; 

			groundTruth[f*m_K+8*i+5] = (mc[f*m_K/8+i] & 4) >> 2; 
			groundTruth[f*m_K+8*i+6] = (mc[f*m_K/8+i] & 2) >> 1; 
			groundTruth[f*m_K+8*i+7] = (mc[f*m_K/8+i] & 1); 
		}
	}
	int errorbit = 0;
	for(int f = 0;f < frame_len; f++){

		for(int i = 0;i < m_K; i++){

			if(result[f*m_K+i] != groundTruth[f*m_K+i]){
				errorbit ++;
			}
		}
	}
	printf( "误码率为：%e\n",(double)errorbit/(frame_len*m_K));


	delete []input;

	delete []result;

	delete []groundTruth;

	return 0;

}

template <class Type>
void print_file(Type *a, int size, string name){

	ofstream ofs;
	ofs.open(name);
	for(int i = 0; i < size; i++){
		ofs << a[i] << ' ';
	}
	ofs << endl;
	ofs.close();

}

void CTCC(int &WaveformNo, int &m_Period, bool *m_Pattern, T &m_Rate, int &m_K, int &m_N, int &m_rlen, int **pi, BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16])
{

	int P, Q0, Q1, Q2, Q3;
	switch (WaveformNo)
	{

	case 32:
		m_Period = 1;
		m_Pattern[0] = 1;
		m_Rate = 1./2;
		m_K = 800;
		P=23, Q0=10, Q1=8, Q2=2, Q3=1; 
		m_rlen = 1600;
		break;
	case 33:
		m_Period = 6;
		m_Pattern[0] = 1;
		m_Pattern[1] = 0;
		m_Pattern[2] = 1;
		m_Pattern[3] = 0;
		m_Pattern[4] = 0;
		m_Pattern[5] = 0;
		m_Rate = 3./4;
		m_K = 800;
		P=23, Q0=10, Q1=8, Q2=2, Q3=1; 
		m_rlen = 1068;
		break;
	case 34:
		m_Period = 1;
		m_Pattern[0] = 1;
		m_Rate = 1./2;
		m_K = 1360;
		P=33, Q0=9, Q1=15, Q2=3, Q3=1; 
		m_rlen = 2720;
		break;
	case 35:
		m_Period=6;
		m_Pattern[0] = 1;
		m_Pattern[1] = 0;
		m_Pattern[2] = 1;
		m_Pattern[3] = 0;
		m_Pattern[4] = 0;
		m_Pattern[5] = 0;
		m_Rate = 3./4;
		m_K = 1360;
		P=33, Q0=9, Q1=15, Q2=3, Q3=1; 
		m_rlen = 1814;
		break;
	case 36:
		m_Period=28;
		memset(m_Pattern, 0, 28);
		m_Pattern[0] = 1;
		m_Pattern[4] = 1;
		m_Pattern[12] = 1;
		m_Pattern[20] = 1;
		m_Rate = 7./8;
		m_K = 1360;
		P=33, Q0=9, Q1=15, Q2=3, Q3=1; 
		m_rlen = 1556;
		break;
	case 37:
		m_Period =2;
		m_Pattern[0] = 1;
		m_Pattern[1] = 0;
		m_Rate = 2./3;
		m_K = 3504;
		P=59, Q0=1, Q1=1, Q2=2, Q3=1; 
		m_rlen = 5256;
		break;


	case 38:
		m_Period =4;
		m_Pattern[0] = 1;
		m_Pattern[1] = 0;
		m_Pattern[2] = 0;
		m_Pattern[3] = 0;

		m_Rate = 4./5;
		m_K = 3504;
		P=59, Q0=1, Q1=1, Q2=2, Q3=1; 
		m_rlen = 4380;
		break;

	case 39:
		m_Period =12;
		m_Pattern[0] = 1;
		m_Pattern[1] = 0;
		m_Pattern[2] = 0;
		m_Pattern[3] = 0;
		m_Pattern[4] = 1;
		m_Pattern[5] = 0;
		m_Pattern[6] = 0;
		m_Pattern[7] = 0;
		m_Pattern[8] = 0;
		m_Pattern[9] = 0;
		m_Pattern[10] = 0;
		m_Pattern[11] = 0;


		m_Rate = 6./7;
		m_K = 3504;
		P=59, Q0=1, Q1=1, Q2=2, Q3=1; 
		m_rlen = 4088;
		break;
	}

	m_N = m_K/2;
	*pi = new int[m_N];
	setinterleaver(*pi, P, Q0, Q1, Q2, Q3,m_N);

	settrellis(nextS, lastS, vPos, nextO);
}

void settrellis(BYTE nextS[][16], BYTE lastS[][16], BYTE vPos[][16], bool nextO[][16])
{
	bool s1, s2, s3, s4;
	bool ns1, ns2, ns3, ns4;
	BYTE m;
	BYTE i;

	for (m=0; m<4; m++)
	{
		bool a=(m>>1);
		bool b=(m&1);
		for(i=0; i<16; i++)
		{
			s1 = (i>>3);
			s2 = (i>>2)&1;
			s3 = (i>>1)&1;
			s4 = (i&1);

			ns1 = a^b^s3^s4;
			ns2 = b^s1;
			ns3 = s2;
			ns4 = b^s3;
			bool y = ns1^s1^s2^s4;


			BYTE tmps = (ns1<<3)|(ns2<<2)|(ns3<<1)|ns4;
			nextS[m][i] = tmps;
			lastS[m][tmps] = i;
			vPos[m][tmps] = (m<<1)|y;
			nextO[m][i] = y;
		}
	}
}

void setinterleaver(int *pi, int P, int Q0, int Q1, int Q2, int Q3, int m_N)
{
	int j, Q;
	for (j=0; j<m_N; j++)
	{
		switch(j&3)
		{
		case 0:
			Q = 0;
			break;
		case 1:
			Q = 4*Q1;
			break;
		case 2:
			Q = 4*Q0*P+4*Q2;
			break;
		case 3:
			Q = 4*Q0*P+4*Q3;
			break;
		}

		pi[j] = (P*j+Q+3)%m_N;
	}
}
