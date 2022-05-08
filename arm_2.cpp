#include <iostream>
#include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <fstream>
using namespace std;
const int n = 200;
float m[n][n];
int NUM_THREADS = 7;
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
    int t_id; //�߳� id
};
//�ź�������
sem_t sem_leader;
sem_t *sem_Divsion = new sem_t[NUM_THREADS - 1];
sem_t *sem_Elimination = new sem_t[NUM_THREADS - 1];
//ˮƽ���ֲ�ʹ��neon
void *threadFunc1(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; ++k)
    {
        // t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
        // ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
        // ���ź���������ͬ����ʽ��ʹ�� barrier
        if (t_id == 0)
        {
            for (int j = k + 1; j < n; j++)
            {
                m[k][j] = m[k][j] * 1.0 / m[k][k];
            }
            m[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
        }
        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Divsion[i]);
        }
        //ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            //��ȥ
            for (int j = k + 1; j < n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
        }
        else
        {
            sem_post(&sem_leader);                // ֪ͨ leader, �������ȥ����
            sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
        }
    }
    pthread_exit(NULL);
}
//ˮƽ����ʹ��neon
void *threadFunc2(void *param)
{
    float32x4_t va, vx, vaij, vaik, vakj, vt;
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; ++k)
    {
        vt = vmovq_n_f32(m[k][k]);
        // t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
        // ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
        // ���ź���������ͬ����ʽ��ʹ�� barrier
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
        else
        {
            sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
        }
        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Divsion[i]);
        }
        //ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS)
        {
            //��ȥ
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
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
        }
        else
        {
            sem_post(&sem_leader);                // ֪ͨ leader, �������ȥ����
            sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
        }
    }
    pthread_exit(NULL);
}
//��ֱ���ֲ�ʹ��neon
void *threadFunc3(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; ++k)
    {
        if (t_id == 0)
        {
            for (int j = k + 1; j < n; j++)
            {
                m[k][j] = m[k][j] * 1.0 / m[k][k];
            }
            m[k][k] = 1.0;
        }
        else
        {
            sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
        }
        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Divsion[i]);
        }
        //ѭ����������
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1 + t_id; j < n; j += NUM_THREADS)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
        }
        else
        {
            sem_post(&sem_leader);                // ֪ͨ leader, �������ȥ����
            sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
        }
    }
    pthread_exit(NULL);
    return NULL;
}
int main()
{
    ofstream out("output.txt");
    struct timeval start, over;
    double timeUse;

    {
        out << n << "\t";
        out << NUM_THREADS << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
        //��ʼ���ź���
        sem_init(&sem_leader, 0, 0);
        for (int i = 0; i < NUM_THREADS - 1; ++i)
        {
            sem_init(sem_Divsion, 0, 0);
            sem_init(sem_Elimination, 0, 0);
        }
        //�����߳�
        pthread_t *handles = new pthread_t[NUM_THREADS];       // ������Ӧ�� Handle
        threadParam_t *param = new threadParam_t[NUM_THREADS]; // ������Ӧ���߳����ݽṹ
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc1, (void *)&param[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            pthread_join(handles[t_id], NULL);
        //���������ź���
        sem_destroy(&sem_leader);
        sem_destroy(sem_Divsion);
        sem_destroy(sem_Elimination);
        gettimeofday(&over, NULL);
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
    }

    {
        out << n << "\t";
        out << NUM_THREADS << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
        //��ʼ���ź���
        sem_init(&sem_leader, 0, 0);
        for (int i = 0; i < NUM_THREADS - 1; ++i)
        {
            sem_init(sem_Divsion, 0, 0);
            sem_init(sem_Elimination, 0, 0);
        }
        //�����߳�
        pthread_t *handles = new pthread_t[NUM_THREADS];       // ������Ӧ�� Handle
        threadParam_t *param = new threadParam_t[NUM_THREADS]; // ������Ӧ���߳����ݽṹ
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc2, (void *)&param[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            pthread_join(handles[t_id], NULL);
        //���������ź���
        sem_destroy(&sem_leader);
        sem_destroy(sem_Divsion);
        sem_destroy(sem_Elimination);
        gettimeofday(&over, NULL);
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
    }

    {
        out << n << "\t";
        out << NUM_THREADS << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
        //��ʼ���ź���
        sem_init(&sem_leader, 0, 0);
        for (int i = 0; i < NUM_THREADS - 1; ++i)
        {
            sem_init(sem_Divsion, 0, 0);
            sem_init(sem_Elimination, 0, 0);
        }
        //�����߳�
        pthread_t *handles = new pthread_t[NUM_THREADS];       // ������Ӧ�� Handle
        threadParam_t *param = new threadParam_t[NUM_THREADS]; // ������Ӧ���߳����ݽṹ
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        {
            param[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc3, (void *)&param[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            pthread_join(handles[t_id], NULL);
        //���������ź���
        sem_destroy(&sem_leader);
        sem_destroy(sem_Divsion);
        sem_destroy(sem_Elimination);
        gettimeofday(&over, NULL);
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
    }
    out.close();
}
