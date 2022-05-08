#include <iostream>
#include <random>
#include <arm_neon.h>
#include <fstream>
#include <sys/time.h>

using namespace std;

//生成不会出错的测试用例
// param:矩阵阶数
float **generateMatrix(int n)
{
    float **m = new float *[n];
    for (int i = 0; i < n; i++)
    {
        m[i] = new float[n];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            m[i][j] = rand() % 10;
        }
    }
    return m;
}

//串行平凡高斯消去算法
// param:矩阵,矩阵阶数
void GE_S_N(float **m, int n)
{
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

//第一处并行优化
// param:矩阵,矩阵阶数
void GE_P_1(float **m, int n)
{
    float32x4_t va, vt;
    for (int k = 0; k < n; k++)
    {
        vt = vmovq_n_f32(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 < n; j += 4)
        {
            va = vld1q_f32(&m[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32((float32_t *)&m[k][j], va);
        }
        for (j; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

//第二处并行优化
// param:矩阵,矩阵阶数
void GE_P_2(float **m, int n)
{
    float32x4_t vaik, vakj, vaij, vx;
    for (int k = 0; k < n; k++)
    {
        int j = 0;
        for (int j = k + 1; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(m[i][k]);
            for (j = k + 1; j + 4 < n; j += 4)
            {
                vakj = vld1q_f32(&m[k][j]);
                vaij = vld1q_f32(&m[i][j]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32((float32_t *)&m[i][j], vaij);
            }
            for (j; j < n; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

//两处并行优化
// param:矩阵,矩阵阶数
void GE_P(float **m, int n)
{
    float32x4_t va, vt, vaik, vakj, vaij, vx;
    for (int k = 0; k < n; k++)
    {
        vt = vmovq_n_f32(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 < n; j += 4)
        {
            va = vld1q_f32(&m[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32((float32_t *)&m[k][j], va);
        }
        for (j; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(m[i][k]);
            for (j = k + 1; j + 4 < n; j += 4)
            {
                vakj = vld1q_f32(&m[k][j]);
                vaij = vld1q_f32(&m[i][j]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32((float32_t *)&m[i][j], vaij);
            }
            for (j; j < n; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

//内存对齐的并行优化
// param:矩阵,矩阵阶数
void GE_P_A(float **m, int n)
{
    float32x4_t va, vt, vaik, vakj, vaij, vx;
    for (int k = 0; k < n; k++)
    {
        vt = vmovq_n_f32(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 < n; j += 4)
        {
            if (j % 4 != 0)
            {
                m[k][j] = m[k][j] / m[k][k];
                j -= 3;
                continue;
            }
            va = vld1q_f32(&m[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32((float32_t *)&m[k][j], va);
        }
        for (j; j < n; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            vaik = vmovq_n_f32(m[i][k]);
            for (j = k + 1; j + 4 < n; j += 4)
            {
                if (j % 4 != 0)
                {
                    m[i][j] -= m[i][k] * m[k][j];
                    j -= 3;
                    continue;
                }
                vakj = vld1q_f32(&m[k][j]);
                vaij = vld1q_f32(&m[i][j]);
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32((float32_t *)&m[i][j], vaij);
            }
            for (j; j < n; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

//主函数进行计时等工作
int main()
{
    ofstream out("output.txt");
    for (int n = 200; n <= 4000; n += 400)
    {
        float **m = generateMatrix(n);
        out << n << "\t";

        //顺序为：串行|优化第一个循环|优化第二个循环|优化两个循环|优化两个循环且对齐
        gettimeofday(&start1, NULL);
        GE_S_N(m, n);
        gettimeofday(&end1, NULL);
        time_use = (end1.tv_sec - start1.tv_sec) * 1000000 + (end1.tv_usec - start1.tv_usec);
        out << time_use / 1000 << "\t";

        gettimeofday(&start2, NULL);
        GE_P_1(m, n);
        gettimeofday(&end2, NULL);
        time_use = (end2.tv_sec - start2.tv_sec) * 1000000 + (end2.tv_usec - start2.tv_usec);
        out << time_use / 1000 << "\t";

        gettimeofday(&start3, NULL);
        GE_P_2(m, n);
        gettimeofday(&end3, NULL);
        time_use = (end3.tv_sec - start3.tv_sec) * 1000000 + (end3.tv_usec - start3.tv_usec);
        out << time_use / 1000 << "\t";

        gettimeofday(&start4, NULL);
        GE_P(m, n);
        gettimeofday(&end4, NULL);
        time_use = (end4.tv_sec - start4.tv_sec) * 1000000 + (end4.tv_usec - start4.tv_usec);
        out << time_use / 1000 << "\t";

        gettimeofday(&start5, NULL);
        GE_P_A(m, n);
        gettimeofday(&end5, NULL);
        time_use = (end5.tv_sec - start5.tv_sec) * 1000000 + (end5.tv_usec - start5.tv_usec);
        out << time_use / 1000 << endl;

        for (int i = 0; i < n; i++)
        {
            delete[] m[i];
        }
        delete[] m;
    }
    out.close();
    return 0;
}
