// Luke Doman
// ldoman
// Assignment 4

#include <stdlib.h>
#include <stdio.h>
#include "linkedListOperations.h"

void mainMenu()
{
  printf("1. Add Element \n");
  printf("2. Search \n");
  printf("3. Insert \n");
  printf("4. Display \n");
  printf("5. Exit \n");
  printf(">>");
}

void main(int argc, char *argv[])
{
  // Init list
  createList();
  
  // Menu
  mainMenu();
  int choice;
  printf("Enter your option:");
  scanf("%d",&choice);
  
  // Main loop for menu choice
  while(choice != 5)
  {
    if(choice == 1)
    {
      int value;
      printf("Enter the number:");
      scanf("%d",&value);

      addElement(value); 
    }
    else if(choice == 2)
    {
      printf("length: %d", length());
      int value;
      printf("Enter the number:");
      scanf("%d",&value);

      search(value);
    }
    else if(choice == 3)
    {
      int value;
      printf("Enter the number:");
      scanf("%d",&value);
      
      insert(value);
    }
    else if(choice = 4)
    {
      display();
    }
    else
    {
      printf("Invalid input.");
    }

    mainMenu();
    scanf("%d",&choice);    
  }
}



