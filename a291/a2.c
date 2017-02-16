// Luke Doman
// ldoman
// Assignment 2

// Command line arguments are expected as... 
// ./A2 10 "*" 10
// ./A2 2 "+" 2

#include <math.h>
#include <stdio.h>

// Convert decimal based int to octal based
int convertToOctal(int decNum)
{
  int oct = 0;
  int n = 0;

  // Change to octal based
  while(decNum >0)
  {
    oct += (pow(10,n++))*(decNum % 8);
    decNum /= 8;
  }
  
  // We need reverse our answer because it is currently smallest bit first
  int answer = 0;

  while (oct != 0)
  {
    answer *= 10;
    answer += oct%10;
    oct /= 10;
  }

  return answer;
}

// Convert octal to binary
int convertToBinary(int octNum)
{
  int bin = 0;
  int dec = 0;
  int n = 0;

  // Change to decimal based
  while(octNum >0)
  {
    dec += (pow(8,n++))*(octNum % 10);
    octNum /= 10;
  }

  //change from dec to binary
  n = 1;
  while(dec > 0)
  {
    bin += (dec%2)*n;
    dec /= 2;
    n *= 10;
  }

  return bin;
}

void main(int argc, char *argv[])
{
  // Read command line arguments
  int n1 =  atoi(argv[1]);
  char func =  (char)argv[2][0];
  int n2 = atoi(argv[3]);

  // Convert operator and operand to octal
  int octalN1 = convertToOctal(n1);
  int octalN2 = convertToOctal(n2);
  
  // Process operation
  int answer = 0;
  if (func == '+')
  {
    answer = octalN1 + octalN2;
  }
  else if(func == '*')
  {
    answer = octalN1 * octalN2;
  }
  else
  {
    printf("Invalid operation entered. %s is not valid operation.", argv[2]);
  }
  
  // Set answer = to binary equivalent
  answer = convertToBinary(answer);
  printf("%d \n", answer);
}
