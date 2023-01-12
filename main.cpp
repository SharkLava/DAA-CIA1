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
		matrix[i][j] = matrix[i-1][j-1] - a;
		if(matrix[i][j]<=0){
			matrix[i][j] = 0;
		}
	}
	else if(matrix[i-1][j]>matrix[i-1][j-1] && matrix[i-1][j]>matrix[i][j-1]){//Down Max
		l++;
		matrix[i][j] = matrix[i-1][j] - (a+(b*(l-1)));
		if(matrix[i][j]<=0){
			matrix[i][j] = 0;
		}
	}
	else if(matrix[i][j-1] > matrix[i-1][j-1] && matrix[i-1][j] > matrix[i][j-1]){//Side Max
		l++;
		matrix[i][j] = matrix[i][j-1] - (a+(b*(l-1)));
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

void prims(int i, int j){
	while(true){
		
		if(matrix[i-1][j-1]>=matrix[i-1][j] && matrix[i-1][j-1]>=matrix[i][j-1]){//Diagonal Max
			al1 = s1[i]+al1;
			al2 = s2[j]+al2;
			i--;
			j--;
		}
		else if(matrix[i-1][j]>matrix[i-1][j-1] && matrix[i-1][j]>matrix[i][j-1]){//Down Max
			al1 = s1[i]+al1;
			al2 = " "+al2;
			i--;
		}
		else if(matrix[i][j-1] > matrix[i-1][j-1] && matrix[i-1][j] > matrix[i][j-1]){//Side Max
			al1 = " "+al1;
			al2 = s2[j]+al2;
			j--;
		}
	}
	std::cout<<al1<<"\n";
	std::cout<<al2<<"\n";
	al1 = "";
	al2 = "";
}


void traceback(){
	int max_x = 1, max_y = 1, max = matrix[1][1];
	for(int i =1;i<6;i++){
		for(int j = 1; j<6;j++){
			if(matrix[i][j]> max){
				max_x = i;
				max_y = j;
				max = matrix[i][j];
			}			
		}
	}



}

int main(){
	fillMatrix(s1,s2);
	printMatrix();
	traceback();
	return 0;
}
