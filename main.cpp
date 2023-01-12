#include <iostream>
#include <string>
#include <stdlib.h>

int matrix[6][6] = {0};
int a = 1, b=2,l=0;
std::string al1,al2;
std::string s1 = " AGTAC";
std::string s2 = " APGTA";
void printMatrix(){
	for (int i = 0; i<6;i++){
		for(int j = 0; j<6;j++){
			std::cout<<matrix[i][j]<<" ";
		}
		std::cout<<"\n";
	}
}

void maxOption(int i, int j){
	if(matrix[i-1][j-1]>=matrix[i-1][j] && matrix[i-1][j-1]>=matrix[i][j-1]){//Diagonal Max
		l=0;
		matrix[i][j] = matrix[i-1][j-1] - (a+(b*l));
		if(matrix[i][j]<=0){
			matrix[i][j] = 0;
		}
	}
	else if(matrix[i-1][j]>matrix[i-1][j-1] && matrix[i-1][j]>matrix[i][j-1]){//Down Max
		l++;
		matrix[i][j] = matrix[i-1][j] - (a+(b*l));
		if(matrix[i][j]<=0){
			matrix[i][j] = 0;
		}
	}
	else if(matrix[i][j-1] > matrix[i-1][j-1] && matrix[i-1][j] > matrix[i][j-1]){//Side Max
		l++;
		matrix[i][j] = matrix[i][j-1] - (a+(b*l));
		if(matrix[i][j]<=0){
			matrix[i][j] = 0;
		}
	}
}

void fillMatrix(std::string s1, std::string s2){
	for(int i = 1;i<6;i++){
		for(int j =1;j<6;j++){
			if(s1[i] == s2[j]){
				matrix[i][j]= matrix[i-1][j-1] + 2;// +2 for match
				if(matrix[i][j]<=0){
					matrix[i][j] = 0;
				}
			}
			else{
				maxOption(i,j);//Affine penalty stuff
			}

		}
	}
}

void traceback(){
	
}

int main(){
	std::cout<<s1<<"\n";
	std::cout<<s2<<"\n";
	fillMatrix(s1,s2);
	printMatrix();
	std::cout<<al1<<"\n";
	std::cout<<al2<<"\n";
	return 0;
}
