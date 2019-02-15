// STL implementation of Prim's algorithm for MST
#include<bits/stdc++.h>
using namespace std;
# define INF 9999999

// iPair ==>  Integer Pair
typedef pair<int, int> iPair;

// This class represents a directed graph using
// adjacency list representation
class Graph
{
    int V; //vertices
    list< pair<int, int> > *adj; //adjacency list for vertex and weight

public:
    Graph(int V);
    void addEdge(int u, int v, int w);
    void primMST();
};

Graph::Graph(int V)
{
    this->V = V;
    adj = new list<iPair> [V]; //allocates graph memory
}

void Graph::addEdge(int u, int v, int w)
{
    adj[u].push_back(make_pair(v, w));
    adj[v].push_back(make_pair(u, w)); //pairs outgoing vertex and weight for edge
}

// Prints shortest paths from src to all other vertices
void Graph::primMST()
{

    int root = 0; // root = 0

    //initalizes keys as infinite, mst array as -1
    //all vertices initally not in queue
    vector<int> key(V, INF);
    vector<int> parent(V, -1);
    vector<bool> inMST(V, false);

    //create pq
    priority_queue< iPair, vector <iPair> , greater<iPair> > pq;
    // Insert source itself in priority queue and initialize
    // its key as 0.
    pq.push(make_pair(0, root));
    key[root] = 0;

    /* Looping till priority queue becomes empty */
    while (!pq.empty())
    {
        //min key vertex = .first
        //min key vertex  weight = .second
        int u = pq.top().second;
        pq.pop(); //get min weight and pop from queue
        inMST[u] = true;  // Include vertex in MST

        // 'i' is used to get all adjacent vertices of a vertex
        list< pair<int, int> >::iterator i;
        for (i = adj[u].begin(); i != adj[u].end(); ++i)
        {
            //get adjacent label and weight
            int v = (*i).first;
            int weight = (*i).second;

            //if v is not already in MST and the edge weight (u,v) < current min
            if (inMST[v] == false && key[v] > weight)
            {
                // Updating key of v
                key[v] = weight; //store the weight in a key array
                pq.push(make_pair(key[v], v));//pair the new smallest weight to v
                parent[v] = u; //(u,v) now new minimum edge in MST
            }
        }
    }

    // Print edges of MST using parent array
    for (int i = 1; i < V; ++i)
        printf("%d - %d\n", parent[i], i);
}

// Driver program to test methods of graph class
int main()
{
    // create the graph given in above fugure
    int V = 9;
    Graph g(V);

    //  making above shown graph
    g.addEdge(0, 1, 4);
    g.addEdge(0, 7, 8);
    g.addEdge(1, 2, 8);
    g.addEdge(1, 7, 11);
    g.addEdge(2, 3, 7);
    g.addEdge(2, 8, 2);
    g.addEdge(2, 5, 4);
    g.addEdge(3, 4, 9);
    g.addEdge(3, 5, 14);
    g.addEdge(4, 5, 4);
    g.addEdge(6, 5, 2);
    g.addEdge(6, 7, 1);
    g.addEdge(6, 8, 4);
    g.addEdge(7, 8, 4);

    g.primMST();

    return 0;
}
