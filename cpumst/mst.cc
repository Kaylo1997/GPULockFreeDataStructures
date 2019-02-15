#include <iostream>
#include <iomanip>
#include <queue>

#define size 5

void mst(int graph[size][size]);
int minimumKey(int key[], bool mstIncluded[]);
void print(int mstGraph[], int graph[size][size]);

int main(){


  int graph[size][size] = {{0, 3, 0, 0, 8},
                          {3, 0, 4, 1, 1},
                          {0, 4, 0, 6, 8},
                          {8, 1, 8, 0, 0},
                          {0, 1, 6, 0, 0}};


  std::list<pair>[size] graph;
  std::list<pair> adjacentTo = graph[0];

  mst(graph);
  return 0;

}

struct pair{
  int index;
  int weight;
  bool operator<(const pair& other) {
    return weight < other.weight;
  }
};

void mst(int graph[size][size]){

  std::priority_queue <int, std::vector<int>, std::greater<int> > min_pq;
  int key[size]; //keys to choose minimum edge
  int mstGraph[size]; //the final mst
  bool mstIncluded[size]; //stores whether vertice is included or not


  for (int i = 0; i< size; i++){
    key[i] = 9999999; //keys initially set to very high number
    mstIncluded[i]=0; //all nodes initially not in mst
    min_pq.push(9999999);
  }


  key[0] = 0;
  mstGraph[0] = -1; //first node root of mst
  min_pq.push(0);


  /*for(int i = 0; i<size-1;i++){//for all vertices in the graph
    int index = minimumKey(key,mstIncluded);//get min index
    mstIncluded[index] = 1;//now included in mst
    for(int j=0; j<size; j++){
      if(graph[index][j] && mstIncluded[j]==false&&graph[index][j]<key[j]){
        mstGraph[j] = index;
        key[j] = graph[index][j];

      }
    }
  }*/

  while(!min_pq.empty()){

    int index = min_pq.top();
    min_pq.pop();

    mstIncluded[index]=true;

    for(int j=0; j<size; j++){
      if(mstIncluded[j]==false){
        mstGraph[j] = index;
        min_pq.push(graph[index][j]);
    }
    }
}
  print(mstGraph, graph);
}

int minimumKey(int key[], bool mstIncluded[]){
  int min = 9999999;
  int indexMin;

  for (int i =0;i<size;i++){ //iterate along all vertices
    if(mstIncluded[i]==0 && key[i]<min){//if the vertice is not already included in mst and the at that value is less than min
      min = key[i];
      indexMin = i;//save min and key index
    }
  }

  return indexMin;
}

void print(int mstGraph[], int graph[size][size]){
  std::cout<<" Edge    Weight\n";
  for(int i = 1;i<size;i++){
    std::cout<<mstGraph[i]<<" - "<<i<<"     "<<graph[i][mstGraph[i]]<<"\n";
  }

}
