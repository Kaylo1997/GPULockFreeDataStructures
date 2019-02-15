#include<bits stdc++.h="">
using namespace std;
# define INF 999.

// iPair ==> Integer Pair
typedef pair<int, double=""> idPair;
typedef pair<double, int=""> diPair;

// This class represents a directed graph using
// adjacency list representation
class Graph
{
int V; // No. of vertices

// In a weighted graph, we need to store vertex
// and weight pair for every edge
list< idPair > *adj;

public:
Graph(int V); // Constructor
~Graph();

// function to add an edge to graph
void addEdge(int u, int v, double w);

// Print MST using Prim's algorithm
void primMST();
};

// Allocates memory for adjacency list
Graph::Graph(int V)
{
this->V = V;
adj = new list<idpair> [V];
}

Graph::~Graph()
{
delete [] adj;
}

void Graph::addEdge(int u, int v, double w)
{
adj[u].push_back(make_pair(v, w));
adj[v].push_back(make_pair(u, w));
}

// Prints shortest paths from src to all other vertices
void Graph::primMST()
{
// Create a priority queue to store vertices that
// are being preinMST. This is weird syntax in C++.
// Refer below link for details of this syntax
// http://geeksquiz.com/implem...
priority_queue< diPair, vector <dipair> , greater<dipair> > pq;

int src = 0; // Taking vertex 0 as source

// Create a vector for keys and initialize all
// keys as infinite (INF)
vector<double> key(V, INF);

// To store parent array which in turn store MST
vector<int> parent(V, -1);

// To keep track of vertices included in MST
vector<bool> inMST(V, false);

// Insert source itself in priority queue and initialize
// its key as 0.
pq.push(make_pair(0.0, src));
key[src] = 0.0;

/* Looping till priority queue becomes empty */
while (!pq.empty())
{
// The first vertex in pair is the minimum key
// vertex, extract it from priority queue.
// vertex label is stored in second of pair (it
// has to be done this way to keep the vertices
// sorted key (key must be first item
// in pair)
int u = pq.top().second;
pq.pop();

inMST[u] = true; // Include vertex in MST

// 'i' is used to get all adjacent vertices of a vertex
list< pair<int, double=""> >::iterator i;
for (i = adj[u].begin(); i != adj[u].end(); ++i)
{
// Get vertex label and weight of current adjacent
// of u.
int v = (*i).first;
double weight = (*i).second;

// If v is not in MST and weight of (u,v) is smaller
// than current key of v
if (inMST[v] == false && key[v] > weight)
{
// Updating key of v
key[v] = weight;
pq.push(make_pair(key[v], v));
parent[v] = u;
}
}
}

// Print edges of MST using parent array
for (int i = 1; i < V; ++i)
printf("%3d - %3d %8.4f\n", parent[i], i, key[i]);
}

// Test client
int main() {
int V = 20;
double w;
Graph g( V );
srand( 17 );

for ( int i = 0; i < V; ++i ) {
for ( int j = i + 1; j < V; ++j ) {
if ( static_cast<double>( rand() ) / RAND_MAX < 0.3 ) {
w = ( static_cast<double>( rand() ) / RAND_MAX ) * 10.;
g.addEdge( i, j, w );
//printf( "%d - %d (%7.3f)\n", i, j, w );
}
}
}

g.primMST();
return 0;
}
