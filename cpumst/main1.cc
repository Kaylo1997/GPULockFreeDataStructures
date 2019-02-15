// STL implementation of Prim's algorithm for MST
#include<bits/stdc++.h>
using namespace std;

using Graph_t = std::vector<GraphNode>;

// Driver program to test methods of graph class
int main()
{
    // TODO build a Graph_t instance here
    Graph_t theGraph(3);
    theGraph[0].addEdge(GraphEdge(2, &theGraph[1]));
    theGraph[1].addEdge(GraphEdge(5, &theGraph[2]));
    theGraph[2].addEdge(GraphEdge(1, &theGraph[0]));
    MST(theGraph);

    return 0;
}

struct GraphEdge {
  int weight;
  GraphNode* dest;
  bool operator<(const GraphEdge& other) const {
    return weight < other.weight;
  }
  GraphEdge(int iweight, GraphNode* idest): weight(iweight), dest(idest) {}
}

struct GraphNode {
  int id;
  std::vector<GraphEdge> outgoingEdges;
  bool isPartOfTree;

  void addEdge(const GraphEdge& edge) {
    outgoingEdges.push_back(edge);
  }

  GraphNode(int iid):id(iid), isPartOfTree(false){}
}

void MST(Graph_t& graph){
  //create the priority queue
  priority_queue<GraphEdge, std::vector<GraphEdge>, greater<GraphEdge> > pq;

  GraphNode* root = &graph.at(0);
  root->isPartOfTree = true;
  for (auto& adjToRoot: root->outgoingEdges) {
    pq.push(adjToRoot);
  }

  std::vector<GraphEdge> edgesInTree;
  while(!pq.empty()) {
    GraphEdge curMinEdge = pq.top();
    pq.pop();
    GraphNode* adjNode = curMinEdge.dest;
    if (!adjNode->isPartOfTree) {
      edgesInTree.push_back(curMinEdge);
      adjNode->isPartOfTree = true;
      for (auto adjNodeEdge: adjNode->outgoingEdges) {
        pq.push(adjNodeEdge);
      }
    }
  }

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
