#include <iostream>
#include <pthread.h>
#include <sys/time.h>
#include <arm_neon.h>
#include <fstream>
using namespace std;

const int n = 200;
float m[n][n];
int worker_count = 3; //工作线程数量
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
    int k;    //消去的轮次
    int t_id; // 线程 id
};

void *threadFunc(void *param)
{
    float32x4_t vx, vaij, vaik, vakj;
    threadParam_t *p = (threadParam_t *)param;
    int k = p->k;         //消去的轮次
    int t_id = p->t_id;   //线程编号
    int i = k + t_id + 1; //获取自己的计算任务
    for (int s = k + 1 + t_id; s < n; s += worker_count)
    {
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
            m[s][j] = m[s][j] - m[s][k] * m[k][j];

        m[s][k] = 0;
    }
    pthread_exit(NULL);
}

int main()
{
    ofstream out("output.txt");
    struct timeval start, over;
    double timeUse;
    for (; worker_count <= 7; worker_count++)
    {
        // for (; n <= 2000; n += 200)
        // {
        out << n << "\t";
        out << worker_count << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
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
            //创建工作线程，进行消去操作
            pthread_t *handles = new pthread_t[worker_count];       // 创建对应的 Handle
            threadParam_t *param = new threadParam_t[worker_count]; // 创建对应的线程数据结构

            //分配任务
            for (int t_id = 0; t_id < worker_count; t_id++)
            {
                param[t_id].k = k;
                param[t_id].t_id = t_id;
            }
            //创建线程
            for (int t_id = 0; t_id < worker_count; t_id++)
            {
                pthread_create(&handles[t_id], NULL, threadFunc, (void *)&param[t_id]);
            }
            //主线程挂起等待所有的工作线程完成此轮消去工作
            for (int t_id = 0; t_id < worker_count; t_id++)
            {
                pthread_join(handles[t_id], NULL);
            }
        }
        gettimeofday(&over, NULL);
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
        // }
    }
    out.close();
    return 0;
}
