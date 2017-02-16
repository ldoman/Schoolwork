// Luke Doman
// ldoman
// Assignment 4

#include <stdlib.h>
#include <stdio.h>
#include "operations.h"

void mainMenu()
{
  printf("1. Update record \n");
  printf("2. Print all student records \n");
  printf("3. Find out average of class \n");
  printf("4. Exit \n");
  printf(">>");
}

void studentMenu()
{
  printf("1. Name \n");
  printf("2. Contact number \n");
  printf("3. Address \n");
  printf("4. Assignment \n");
  printf("5. Midterm \n");
  printf("6. Final \n");
  printf("7. Go back \n");
  printf(">>");
}

void main(int argc, char *argv[])
{
  int numStudents;
  printf("Enter number of students to be created:");
  scanf("%d",&numStudents);

  generateStudents(numStudents);

  int choice = 0;
  int student = 0;
  int studentChoice = 0;

  mainMenu();
  scanf("%d",&choice);
  
  while(choice != 4)
  {
    if(choice == 1)
    {
      printf("Enter the student number:");
      scanf("%d",&student);

      studentMenu();
      scanf("%d", &studentChoice);
      
      while(studentChoice != 7)
      {
	update(student,studentChoice);
	studentMenu();
	scanf("%d", &studentChoice);
      }
    }
    else if(choice == 2)
    {
      printAll();
    }
    else if(choice == 3)
    {
      averages();
    }
    else
    {

    }

    mainMenu();
    scanf("%d",&choice);    
  }
}



