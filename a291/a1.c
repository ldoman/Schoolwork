// Luke Doman
// ldoman
// Assignment 1

#include <math.h>
#include <stdio.h>

void main()
{
  int n = 8208;
  int numSize = (int)floor(log10(abs(n))) + 1;
  
  if(numSize == 1)
  {
    printf("The number is an Armstrong number. \n");
  }
  else
  {
    int sum = 0;
    int index = numSize;
    int num = n;
    int currentNum = 0;

    while(index != 0)
    {
      currentNum = num%10;
      sum += (int) pow((double) currentNum, numSize);
      num = num/10;
      index--;
    }
    
    if(sum == n)
    {
      printf("The number is an Armstrong number. \n");
    }
    else
    {
      printf("The number is not an Armstrong number. \n");
    }
  }
}
