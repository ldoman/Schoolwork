// Luke Doman
// ldoman
// Assignment 3

// This will only work for < 20 command line arguments

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

void main(int argc, char *argv[])
{
  // Array to store cla ints
  int nums[20] = {0};
 
  // Read command line arguments
  for(int i = 0; i < argc-1; i++)
  {
     nums[i] = atoi(argv[i+1]);
  }

  // Variabes we need for finding max product
  int maxProduct = INT_MIN;
  int tempProduct = 0;
  int witnessIndex1 = 0;
  int witnessIndex2 = 0;

  // Iterate over nums and test all different combinations...O(n^2)
  for(int i = 0; i < 20; i++)
  {
    for(int j = 0; j < 20; j++)
    {
      // Skip multiplication if we see a 0
      if(nums[i] == 0 || nums[j] == 0)
      {
		tempProduct = 0;
      }
      else if (i == j)
      {
		continue;
      }
      else
      {
		tempProduct = nums[i] * nums[j];
      }
      
      // If temp is larger than max set max equal to the new max and update witnesses
      if(tempProduct > maxProduct)
      {
		maxProduct = tempProduct;
        witnessIndex1 = i;
		witnessIndex2 = j;
      }
    }
  }

  printf("%d %d \n", nums[witnessIndex1], nums[witnessIndex2]);
}
