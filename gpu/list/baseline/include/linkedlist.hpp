/*

Copyright 2012-2013 Indian Institute of Technology Kanpur. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions, and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY INDIAN INSTITUTE OF TECHNOLOGY KANPUR ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INDIAN INSTITUTE OF TECHNOLOGY KANPUR OR
THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied, of Indian Institute of Technology Kanpur.

*/

/**********************************************************************************

 Lock-free linked list for CUDA; tested for CUDA 4.2 on 32-bit Ubuntu 10.10 and 64-bit Ubuntu 12.04.
 Developed at IIT Kanpur.

 Inputs: Percentage of add and delete operations (e.g., 30 50 for 30% add and 50% delete)
 Output: Prints the total time (in milliseconds) to execute the the sequence of operations

 Compilation flags: -O3 -arch sm_20 -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ -DNUM_ITEMS=num_ops -DFACTOR=num_ops_per_thread -DKEYS=num_keys

 NUM_ITEMS is the total number of operations (mix of add, delete, search) to execute.

 FACTOR is the number of operations per thread.

 KEYS is the number of integer keys assumed in the range [10, 9+KEYS].
 The paper cited below states that the key range is [0, KEYS-1]. However, we have shifted the range by +10 so that
 the head sentinel key (the minimum key) can be chosen as zero. Any positive shift other than +10 would also work.

 The include path ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ is needed for cutil.h.

 Related work:

 Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of Concurrent Lock-free Data Structures
 on GPUs. In Proceedings of the 18th IEEE International Conference on Parallel and Distributed Systems,
 December 2012.

***************************************************************************************/
#ifndef LINKEDLIST_H
#define LINKEDLIST_H

//#include"cutil.h"		// Comment this if cutil.h is not available
#include<cuda_runtime.h>
#include<stdio.h>
#include"../../include/cuda_intrinsics.h"

#if __WORDSIZE == 64
typedef unsigned long long LL;
#else
typedef unsigned int LL;
#endif

// Number of threads per block
#define NUM_THREADS 64

// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

// Definition of generic node class

class __attribute__((aligned (16))) Node
{
  public:
    LL key;
    LL next;

    // Create a next field from a reference and mark bit
    __device__ __host__ LL CreateRef(Node* ref, bool mark)
    {
      LL val=(LL)ref;
      val=val|mark;
      return val;
    }

    __device__ __host__ void SetRef(Node* ref, bool mark)
    {
      next=CreateRef(ref, mark);
    }

    // Extract the reference from a next field
    __device__ Node* GetReference()
    {
      LL ref=next;
      return (Node*)((ref>>1)<<1);
    }

    // Extract the reference and mark bit from a next field
    __device__ Node* Get(bool* marked)
    {
      marked[0]=next%2;
      return (Node*)((next>>1)<<1);
    }

    // CompareAndSet wrapper
    __device__ bool CompareAndSet(Node* expectedRef, Node* newRef, bool oldMark, bool newMark)
    {
      LL oldVal = (LL)expectedRef|oldMark;
      LL newVal = (LL)newRef|newMark;
      LL oldValOut=atomicCAS(&(next), oldVal, newVal);
      if (oldValOut==oldVal) return true;
      return false;
    }

    // Constructor for sentinel nodes
    Node(LL k)
    {
      key=k;
      next=CreateRef((Node*)NULL,false);
    }
};

__device__ Node** nodes;			// Pool of pre-allocated nodes
__device__ unsigned int pointerIndex=0;		// Index into pool of free nodes

// Function for creating a new node when requested by an add operation

__device__ Node* GetNewNode(LL key)
{
  LL ind=atomicInc(&pointerIndex, NUM_ITEMS);
  Node* n=nodes[ind];
  n->key=key;
  n->SetRef(NULL, false);
  return n;
}

// Window of node containing a particular key

class Window
{
  public:
    Node* pred;		// Predecessor of node holding the key being searched
    Node* curr;		// The node holding the key being searched (if present)

    __device__ Window(Node* myPred, Node* myCurr)
    {
      pred=myPred;
      curr=myCurr;
    }
};

// Lock-free linked list

class LinkedList
{
  public:
    __device__ void Find(Window*, LL);			// Helping method
    __device__ bool Add(LL);
    __device__ bool Delete(LL);
    __device__ bool Search(LL);

    Node* head;
    Node* tail;

    LinkedList()
		{
			Node* h=new Node(0);				// Head sentinel
#if __WORDSIZE == 64
			Node* t=new Node((LL)0xffffffffffffffff);           // Tail sentinel
#else
			Node* t=new Node((LL)0xffffffff);			// Tail sentinel
#endif

			CHECK_CUDA_ERROR(cudaMalloc((void**)&head, sizeof(Node)));
			CHECK_CUDA_ERROR(cudaMalloc((void**)&tail, sizeof(Node)));
		
			h->next=(LL)tail;
		
			CHECK_CUDA_ERROR(cudaMemcpy(head, h, sizeof(Node), cudaMemcpyHostToDevice));
			CHECK_CUDA_ERROR(cudaMemcpy(tail, t, sizeof(Node), cudaMemcpyHostToDevice));
		}
};

// Find the window holding key
// On the way clean up logically deleted nodes (those with set marked bit)

__device__ void
LinkedList::Find(Window* w, LL key)
{
  Node* pred;
  Node* curr;
  Node* succ;
  bool marked[]={false};
  bool snip;

  retry:
  while(true){
     pred=head;
     curr=pred->GetReference();
     while(true){
        succ=curr->Get(marked);
        while(marked[0]){
           snip=pred->CompareAndSet(curr, succ, false, false);
           if(!snip) goto retry;
	   curr=succ;
	   succ=curr->Get(marked);
        }
        if(curr->key >= key){
           w->pred=pred;
           w->curr=curr;
           return;
	}
	pred=curr;
	curr=succ;
     }
  }
}

__device__ bool 
LinkedList::Search(LL key)
{
  bool marked;
  Node* curr = head;
  while(curr->key<key){
     curr=curr->GetReference();
     Node* succ = curr->Get(&marked);
  }
  return((curr->key == key) && !marked);
}
   
__device__ bool
LinkedList::Delete(LL key)
{
  Window w(NULL, NULL);
  bool snip;
  while(true){
     Find(&w, key);
     Node* curr=w.curr;
     Node* pred=w.pred;
     if(curr->key!=key){
        return false;
     }
     else{
        Node* succ = curr->GetReference();
        snip=curr->CompareAndSet(succ, succ, false, true);
	if(!snip) continue;
	pred->CompareAndSet(curr, succ, false, false);
	return true;
     }
  }
}

__device__ bool
LinkedList::Add(LL key)
{
  Node* pointer=GetNewNode(key);
  Window w(NULL, NULL);
  while(true){
     Find(&w, key);
     Node* pred=w.pred;
     Node* curr=w.curr;
     if (curr->key == key) return false;
     pointer->key=key;
     pointer->SetRef(curr, false);
     bool test=(pred->CompareAndSet(curr, pointer, false, false));
     if(test) return true;
  }
}

__device__ LinkedList* list;            // The linked list

// Kernel for initializing device memory

__global__ void init(LinkedList* List)
{
  list=List;
}

#endif /* LINKEDLIST_H */

