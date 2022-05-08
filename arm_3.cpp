#include <iostream>
#include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <fstream>
using namespace std;
const int n = 200;
float m[n][n];
int NUM_THREADS = 3;
void generateMatrix()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            m[i][j] = rand() % 100;
    }

    for (int i = 0; i < n; i++)
    {
        int k1 = rand() % n;
        int k2 = rand() % n;
        for (int j = 0; j < n; j++)
        {
            m[i][j] += m[0][j];
            m[k1][j] += m[k2][j];
        }
    }
}
struct threadParam_t
{
    int t_id; //线程 id
};
// barrier 定义
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
//线程函数定义
void *threadFunc(void *param)
{
    float32x4_t va, vx, vaij, vaik, vakj, vt;
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; ++k)
    {
        vt = vmovq_n_f32(m[k][k]);
        if (t_id == 0)
        {
            int j;
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                va = vld1q_f32(&(m[k][j]));
                va = vdivq_f32(va, vt);
                vst1q_f32(&(m[k][j]), va);
            }
            for (; j < n; j++)
            {
                m[k][j] = m[k][j] * 1.0 / m[k][k];
            }
            m[k][k] = 1.0;
        }
        //第一个同步点
        pthread_barrier_wait(&barrier_Divsion);
        //循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            //消去
            vaik = vmovq_n_f32(m[i][k]);
            int j;
            for (j = k + 1; j + 4 <= n; j += 4)
            {
                vakj = vld1q_f32(&(m[k][j]));
                vaij = vld1q_f32(&(m[i][j]));
                vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&m[i][j], vaij);
            }
            for (; j < n; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];

            m[i][k] = 0;
        }
        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
}
int main()
{
    ofstream out("output.txt");
    struct timeval start, over;
    double timeUse;
    for (; NUM_THREADS <= 7; NUM_THREADS++)
    {
        out << n << "\t";
        out << NUM_THREADS << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
        //初始化barrier
        pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
        pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);
        //创建线程
        pthread_t *handles = new pthread_t[NUM_THREADS];       // 创建对应的 Handle
        threadParam_t *param = new threadParam_t[NUM_THREADS]; // 创建对应的线程数据结构
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc, (void *)&param[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            pthread_join(handles[t_id], NULL);

        //销毁所有的 barrier
        pthread_barrier_destroy(&barrier_Divsion);
        pthread_barrier_destroy(&barrier_Elimination);
        gettimeofday(&over, NULL); //结束计时
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
    }
    out.close();
}
