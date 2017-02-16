// Luke Doman
// ldoman
// Assignment 5

#include <stdlib.h>
#include <stdio.h>
#include "node.h"

typedef struct node Node;

struct node *HEAD;
struct node list;

void createList()
{
  list.value = 0;
  list.next = malloc(sizeof(Node));
  list.next = NULL;
  HEAD = malloc(sizeof(Node));
  HEAD = &list;
}

// Return length of list
int length()
{
  int count = 0;

  while (list.next != NULL)
  {
    count++;
    list = *list.next;
  }

  list = *HEAD;
}

// Append element to end of list
void addElement(int n)
{
  if(length() == 0)
  {
    list.value = n;
  }
  else
  {
    // Get last node of list
    while (list.next != NULL)
      {
		list = *list.next;
      }

    // Make node from passed data
    struct node temp;
    temp.value = n;
    temp.next = malloc(sizeof(Node));
    temp.next = NULL;

    // Update tail of curent list
    list.next = &temp;
    list = *HEAD;
  }
}

// Return index of passed element
int search(int n)
{
  int index = 0;

  // Iterate over list until it finds the number
  while (list.next != NULL)
  {
    if(list.value == n)
    {
      list = *HEAD;
      return index;
    }

    index++;
    list = *list.next;
  }
  
  // Return -1 if vlaue not found
  list = *HEAD;
  return -1;
}

// Insert elemnet at passed index
void insert(int n)
{
  int index = 0;
  struct node newNode;
  newNode.value = n;

  // Iterate over list until it finds the index
  while (list.next != NULL)
  {
    if(index == n)
    {
      // Store pointer to next node
      newNode.next = list.next;
      
      // Update pointer to new node
      list.next = &newNode;
      
      list = *HEAD;
    }
	
    index++;
    list = *list.next;
  }
  
  list = *HEAD;
}

// List contents
void display()
{
  // Iterate over list
  do 
  {
    printf("%d ->", list.value);
    list = *list.next;
  } while (list.next != NULL);

  list = *HEAD;
}
