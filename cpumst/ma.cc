// STL implementation of Prim's algorithm for MST
#include<bits/stdc++.h>
using namespace std;

//Graph class
class Graph
{
    int size; //vertices
    list< pair<int, int> > *adj; //stores vertex and weight
    std::vector<GraphNode> graphNodes;

public:
    Graph(int size);  // Constructor
    void addEdge(int u, int v, int w); //adds edges
    void MST(); //calls MST and prints
};

// Driver program to test methods of graph class
int main()
{
    Graph g; //initialize graph

    //add edges to graph (node 1, node 2, weight)
    g.addEdge(1, 3, 2);
    g.addEdge(1, 4, 1);
    g.addEdge(3, 4, 3);
    g.addEdge(4, 2, 4);
    g.addEdge(4, 6, 5);
    g.addEdge(5, 4, 22);
    g.addEdge(5, 7, 33);
    g.addEdge(7, 2, 9);
    g.addEdge(2, 6, 11);

    g.MST();

    return 0;
}

Graph::Graph(int size)
{
    this->size = size;
    adj = new list<pair<int, int> > [size];//creates array of pairs of size of graph
}

void Graph::addEdge(int u, int v, int w)
{
  
}

struct GraphEdge {
  int weight;
  GraphNode* dest;
  bool operator<(const GraphEdge& other) const {
    return weight < other.weight;
  }
}

struct GraphNode {
  unsigned int id;
  std::vector<GraphEdge> outgoingEdges;

}


void Graph::MST(){


  //create the priority queue
  priority_queue< pair<int, int>, vector <pair<int, int> > , greater<GraphEdge> > pq;

  int src = 0; // vertex 0 as source
  int infinite = 999999999; //infinity



  // print MST
  for (int i = 1; i < size; ++i)
      printf("%d - %d\n", parent[i], i);
}

/*// Prints shortest paths from src to all other vertices
void Graph::MST()
{

    //create the priority queue
    priority_queue< pair<int, int>, vector <pair<int, int> > , greater<GraphEdge> > pq;

    int src = 0; // vertex 0 as source
    int infinite = 999999999; //infinity

    vector<int> key(size, infinite);//all keys initially infinite
    vector<int> parent(size, -1); //parent array
    vector<bool> mstIncluded(size, false); //all nodes initially not yet in MST

    pq.push(make_pair(0, src)); //create source and insert into pq
    key[src] = 0; //source has key = 0

    //while the queue is still not empty
    while (!pq.empty())
    {
        int minkey = pq.top().second; //since using pq, get minimum key
        pq.pop(); //pop out the top most pair for next iteration
        mstIncluded[minkey] = true;  // minkey index now included in graph

        list< pair<int, int> >::iterator i; //adjaceny list iterator
        for (i = adj[minkey].begin(); i != adj[minkey].end(); ++i)
        {
            int v = (*i).first;
            int weight = (*i).second; //gets vertex and weight of adjacent list of vertexes

            if (mstIncluded[v] == false && key[v] > weight) //if its not mst and weight is smaller than current key
            {
                // Update key of v as new smalles weight
                key[v] = weight;
                pq.push(make_pair(key[v], v));
                parent[v] = minkey;
            }
        }
    }

    // print MST
    for (int i = 1; i < size; ++i)
        printf("%d - %d\n", parent[i], i);
}
*/
