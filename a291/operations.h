
// Luke Doman
// ldoman
// Assignment 4

#include <stdlib.h>
#include <stdio.h>
#include "student.h"

static int numStudents;
static struct personal personalData[10];
static struct uni uniData[10];

void generateStudents(int n)
{
  // Update our reference for count of students
  numStudents = n;

  for(int i = 0; i < numStudents; i++)
  {
    // Init structs with blank values
    personalData[i].name = "";
    personalData[i].phone = 0;
    personalData[i].address = "";
    
    uniData[i].num = i;
    uniData[i].assignment = 0;
    uniData[i].midterm = 0;
    uniData[i].final = 0;
    uniData[i].total = 0;
  }

  printf("%d student(s) created. \n", n);
}

void update(int student, int recordType)
{
  if(recordType == 1)
  {
    printf("Enter student name:");
    scanf("%s", personalData[student].name);
  }
  else if(recordType == 2)
  {
    printf("Enter contact:");
    scanf("%d", personalData[student].phone);
  }
  else if(recordType == 3)
  {
    printf("Enter address:");
    scanf("%s", personalData[student].address);
  }
  else if(recordType == 4)
  {
    printf("Enter assignment grade:");
    scanf("%d", uniData[student].assignment);
  }
  else if(recordType == 5)
  {
    printf("Enter midterm grade:");
    scanf("%d", uniData[student].midterm);
  }
  else if(recordType == 6)
  {
    printf("Enter final name:");
    scanf("%d", uniData[student].final);
  }
  else
  {
    printf("Invalid input.");
  }
}

void printAll()
{
  for(int i = 0; i < 10; i++)
  {
    if(uniData[i].num == NULL)
    {
      continue;
    }

    printf("Student: %s \n", personalData[i].name ); 
    printf("ID: %d \n", uniData[i].num);
    printf("Phone: %d \n", personalData[i].phone); 
    printf("Address: %s \n", personalData[i].address); 
    printf("Assignment Grade: %d \n", uniData[i].assignment); 
    printf("Midterm: %d \n", uniData[i].midterm); 
    printf("Final: %d \n\n", uniData[i].final); 
  }  
}

void averages()
{
  int count = 0;
  double sum = sum;
  
  for(int i = 0; i < 10; i++)
  {
    if(uniData[i].num == NULL)
    {
      continue;
    }

    int stdAvg = uniData[i].assignment + uniData[i].midterm + uniData[i].final;
    count++;
    sum += stdAvg;
  }  

  double rtn = sum/count;
  printf("Average: %d \n",rtn);  
}


