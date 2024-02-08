#include<iostream>
#include<cstring>
#include<cuda.h>
#include<fstream>
#include<thrust/device_vector.h>

#define BLOCKSIZE 512

using namespace std;

//Graph 
  bool *adjacencyMatrix;

__global__ void kclique(int* d_degree, int presentNodes, int K, int N,bool* adjacencyMat,int *d_count)
{
    
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(id < N)
    {
        int st[10000][4];
        int top=-1;

        int dclique[10000];
       
        top++;
        
        dclique[1]=id+1;
     
        st[top][0]=id+2;
        st[top][1]=1;
        st[top][2]=2;
        st[top][3]=K;
        
        d_count[0]=0;
     
        while(top!=-1)
        {
            int j=st[top][0];
            int i=st[top][1];
            int l=st[top][2];
            int s=st[top][3];

            top--;
            if(j+1<=N)
            {
                top++;
             
                st[top][0]=j+1;
                st[top][1]=i;
                st[top][2]=l;
                st[top][3]=s;
            }
         
            if(d_degree[j]>=s-1)
            {
                dclique[l]=j;

                //check if the vertices form a clique or not
                bool flag=true;
             

                //Run a loop for all the set of edges for a vertex
                for(int x=1;x<l+1;x++)
                {
                    for(int y=x+1;y<l+1;y++)
                    {
                        if(adjacencyMat[ 1ll*dclique[x]*1000000+dclique[y] ]==false)
                        {    
                            flag=false;
                            break;
                        }
                    }
                    if(!flag)
                        break;
                }

                //flag will be true if it is clique
                if(flag)
                {
                    if(l<s)
                    {
                        top++;
                     
                        st[top][0]=j+1;
                        st[top][1]=j+1;
                        st[top][2]=l+1;
                        st[top][3]=s;
                    }
                 
                 //If it is clique then atomically increase the count array
                    else
                    {
                       atomicInc((unsigned int*) &d_count[0], 1000000);
                    }
                }
            }
        }
    }
    __syncthreads();
}

int main(int argc, char *argv[])
{
    int k;

    //Degree of the vertices
    int *degree;

    string path = argv[1];
    cin>>k;
    
    //path = "/content/drive/MyDrive/GPU_CP/graph.txt";
    //k = 3;

    degree = (int*)malloc(sizeof(int)*1000000);

    //adjacencyMatrix = (bool*)malloc(sizeof(bool)*1000000000000);
    //bool *d_adjacencyMatrix;

    cudaMallocManaged(&adjacencyMatrix, 1000000000000*sizeof(bool));
    
    ifstream MyReadFile(path);
    string myText;
    int n=0;
    while (getline (MyReadFile, myText)){
        int a,b,i=0;
        string t="";
        while(myText[i]!=' ')
        {
            t+=myText[i];
            i++;
        }
        a=stoi(t);
        b=stoi(myText.substr(i+1));

        n = max(n,max(a,b));

        adjacencyMatrix[1ll*a*1000000+b] = true;
        adjacencyMatrix[1ll*b*1000000+a] = true;
        degree[a]++;
        degree[b]++;
    }



    int count[1];

    int *d_degree,*d_count;
    cudaMalloc((void**)&d_degree, 10000*sizeof(int));
    cudaMalloc((void**)&d_count, 1*sizeof(int));
    
    cudaMemcpy(d_degree, degree, 10000*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, count, 1*sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time = 0.0f;
    
    cudaEventRecord(start, 0); 

    int blocks = 41;

    kclique<<<blocks,BLOCKSIZE>>>(d_degree,1,k,n,adjacencyMatrix,d_count);
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    cudaMemcpy(count, d_count, 1*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cout<<"Number of cliques of size "<<k<<" in the given graph are "<<count[0]<<endl;
    cout<<"Execution Time: "<<gpu_time<<" ms"<<endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_degree);
    cudaFree(d_count);

    free(degree);

    return 0;
}
