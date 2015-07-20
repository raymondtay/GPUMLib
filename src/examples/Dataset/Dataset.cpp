/*
	Ricardo Quintas is an MSc Student at the University of Coimbra, Portugal
    Copyright (C) 2009, 2010 Ricardo Quintas

	This file is part of GPUMLib.

    GPUMLib is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Dataset.h"

double randDouble(double low, double high)
{
	double temp;

	if (low > high)
	{
		temp = low;
		low = high;
		high = temp;
	}

	temp = (rand() / (static_cast<double>(RAND_MAX) + 1.0)) * (high - low) + low;

	return temp;
}

/*Calculate the root mean square error of outputs against targets*/
double rmse_error(HostMatrix<float> &Target,HostMatrix<float> &Output){

	//check global error
	float sum = 0;
	int i;

	for(i = 0; i < Target.Rows(); i++){

		sum = sum + pow(Target(i,0) - Output(i,0),2);		

	}

	return sqrt(sum/Target.Rows());
}



/*Count number of miscalculated outputs against targets*/
int error_calc(HostMatrix<float> &Target,HostMatrix<float> &Output){

	//check global error

	int i;

	int error = 0;

	for(i = 0; i < Target.Rows(); i++){

		if(Target(i,0) - Output(i,0) != 0){
			error += 1;
		}

	}

	return error;
}

/*Read file into matrix X for features and Y for targets, also attributes numbers to classes and 
creates a reverse class lookup to retrieve the names*/
void readFile(string filename, HostMatrix<float> &X, HostMatrix<float> &Y, std::map<string,int> &Classes,std::map<int,string> &ClassesLookup){

	int row,col,num_classes;

	num_classes = 1;

	int N=0;
	int M=0;


	//read input and target;
	ifstream inFile;
	inFile.open(filename.c_str());
	if (!inFile) {
		cout << "Unable to open file";
		exit(1);
	}

	string st;
	string aux;

	if( N == 0 && M == 0){

		int size = 0;
		int size2 = 0;

		while ( !inFile.eof() ){
			getline(inFile, st);

			if(st.empty())
				break;
			
			size++;
		}
		inFile.clear() ; inFile.seekg(0); 
	
		getline(inFile, st);
	
		std::istringstream iss(st);
		while(iss >> aux){
			size2++;
		}

		M = size;
		N = size2 - 1;

	}

	X = HostMatrix<float>(M,N); 
	Y = HostMatrix<float>(M,1);

	row = 0;
	col = 0;

	inFile.clear() ; inFile.seekg(0);
	while (!inFile.eof()) {
		getline(inFile,st);

		if(st.empty())
			break;

		std::istringstream iss(st);

		while(iss >> aux){

			if(col == N){

				if(Classes[aux] == 0){ 
					ClassesLookup[num_classes] = aux;
					Classes[aux] = num_classes++;
				}

				Y(row,0) = Classes[aux];

			}else{
				X(row,col) = atof(aux.c_str());
			}

			col++;
		}

		col = 0;
		row++;
	}

	inFile.close();

}

/*Read file into matrix X for features and Y for targets*/
void readFile(string filename, HostMatrix<float> &X, HostMatrix<float> &Y){

	int row,col;


	int N=0;
	int M=0;

	//read input and target;
	ifstream inFile;
	inFile.open(filename.c_str());
	if (!inFile) {
		cout << "Unable to open file";
		exit(1);
	}

	string st;
	float aux;

	if( N == 0 && M == 0){

		int size = 0;
		int size2 = 0;

		while ( !inFile.eof() ){
			getline(inFile, st);
			
			if(st.empty())
				break;
			
			size++;
		}
		inFile.clear() ; inFile.seekg(0); 

		getline(inFile, st);
		std::istringstream iss(st);
		while(iss >> aux){
			size2++;
		}

		M = size;
		N = size2 - 1;

	}

	X = HostMatrix<float>(M,N); 
	Y = HostMatrix<float>(M,1);

	row = 0;
	col = 0;

	inFile.clear() ; inFile.seekg(0); 
	while (!inFile.eof()) {
		getline(inFile,st);
		
		if(st.empty())
			break;
			
		std::istringstream iss(st);

		while(iss >> aux){

			if(col == N){
				Y(row,0) = aux;
			}else{
				X(row,col) = aux;
			}

			col++;
		}

		col = 0;
		row++;
	}

	inFile.close();

}

/*Normalizes input matrix; Ni = (Xi - Mean) / Std*/
void normalize(HostMatrix<float> &X){

	HostMatrix<float> normalized(X.Rows(),X.Columns());

	for(int j = 0; j < X.Columns(); j++){

		double mean = 0;
		double std = 0;

		double sum = 0;
		for(int i = 0; i < X.Rows(); i++){
			sum = sum + X(i,j);		
		}

		mean = sum/X.Rows();

		sum = 0;
		for(int i = 0; i < X.Rows(); i++){
			sum = sum + pow(X(i,j) - mean,2);
		}

		std = sqrt(sum/(X.Rows()-1));

		for(int i = 0; i < X.Rows(); i++){

			normalized(i,j) = (X(i,j) - mean)/std;

		}

	}
	
	X = normalized;

}

/*Retrieves matrix to be used as test using holdout method, 1/3 for test 2/3 for training*/
HostMatrix<float> holdOutTest(HostMatrix<float> &X){

	HostMatrix<float> output(X.Rows()/3,X.Columns());

	int n = 0;

	for(int i = 2; i < X.Rows(); i+=3){

		for(int j = 0; j < X.Columns(); j++){
			output(n,j) = X(i,j);
		}

		n++;
	}

	return output;

}

/*Retrieves matrix to be used as train using holdout method, 1/3 for test 2/3 for training*/
HostMatrix<float> holdOutTrain(HostMatrix<float> &X){

	HostMatrix<float> output(2*(X.Rows()/3)+1,X.Columns());

	int k = 2;
	int n = 0;

	for(int i = 0; i < X.Rows();){

		for(int j = 0; j < X.Columns(); j++){
			output(n,j) = X(i,j);
		}

		i++;

		if(i == k){
			i++;
			k += 3;
		}

		n++;
	}

	return output;

}

/*Retrieves matrix to be used as test using crossvalidation method*/
HostMatrix<float> crossvalidationTest(HostMatrix<float> &X, int folds, int fold_number){

	int adjust = 0;
	if(fold_number <= X.Rows()%folds){
		adjust += 1;
	}

	HostMatrix<float> output(X.Rows()/folds+adjust,X.Columns());

	int n = 0;

	for(int i = fold_number-1; i < X.Rows(); i+=folds){

		for(int j = 0; j < X.Columns(); j++){
			output(n,j) = X(i,j);
		}

		n++;
	}

	return output;

}

/*Retrieves matrix to be used as train using crossvalidation method*/
HostMatrix<float> crossvalidationTrain(HostMatrix<float> &X, int folds, int fold_number){

	int adjust = 0;
	if(fold_number <= X.Rows()%folds){
		adjust += 1;
	}

	HostMatrix<float> output(X.Rows()-((X.Rows()/folds)+adjust),X.Columns());

	int k = fold_number-1;
	int n = 0;

	for(int i = 0; i < X.Rows();){

		if(i == k){ 
			i++;
			k += folds;
		}

		for(int j = 0; i < X.Rows() && j < X.Columns(); j++){
			output(n,j) = X(i,j);
		}

		i++;
		n++;
	}

	return output;

}


void measures(int correct_instances,int total_instances, int incorrect_instances, int** confusionMatrix, std::map<string,int> Classes, std::map<int,string> ClassesLookup, ofstream &XMLOutput){

	int i,j;

	XMLOutput << "\t<output>" << std::endl;


	/*************SUMMARY************/
	cout << "\n=== Summary ===\n\n";
	cout << std::left << setw(50) << "Correctly Classified Instances " << std::left << setw(5) << correct_instances  << std::right << setw(5) << " " << ((float)correct_instances)/total_instances*100 << "%" << endl;
	cout << std::left << setw(50) << "Incorrectly Classified Instances " << std::left << setw(5) << incorrect_instances << std::right << setw(5) <<" " << ((float)incorrect_instances)/total_instances*100 << "%" <<  endl;
	cout << std::left << setw(50) << "Total Number of Instances " << std::left << setw(5) << total_instances << endl;

	//XMLOutput << ((int)(((float)correct_instances)/total_instances*100));
	
	XMLOutput << "\t\t<correct per=\"" << ((float)correct_instances)/total_instances*100 << "\">" << correct_instances << "</correct>" << std::endl;
	XMLOutput << "\t\t<incorrect per=\"" << ((float)incorrect_instances)/total_instances*100 << "\">" << incorrect_instances << "</incorrect>" << std::endl;

	/********CONFUSION MATRIX*********/
	cout << "\n=== Confusion Matrix ===\n\n";

	XMLOutput << "\t\t<confusion>" << std::endl;

	for(i = 0; i < Classes.size(); i++){
		cout << std::right << setw(5) << ((char) (i+97));
	}
	cout << "  <-- classified as " << endl;


	for(i = 0; i < Classes.size(); i++){
		
		XMLOutput <<	"\t\t\t<class name=\"" << ClassesLookup[i+1] << "\" id=\"" << ((char) (i+97)) << "\" ";
		
		for(j = 0; j < Classes.size(); j++){
			cout << std::right << setw(5) << confusionMatrix[i][j];
			
			XMLOutput << ((char) (j+97)) << "=\"" << confusionMatrix[i][j] << "\" ";
		}
		cout << std::right << setw(5) << " | " << ((char) (i+97)) << " = " << ClassesLookup[i+1] << endl;
		
		XMLOutput << "></class>" << std::endl;
	}
	
	
	XMLOutput << "\t\t</confusion>" << std::endl;
	
	/********************************/

	int *TP = (int*) malloc(sizeof(int)*Classes.size());
	int *TN = (int*) malloc(sizeof(int)*Classes.size());
	int *FP = (int*) malloc(sizeof(int)*Classes.size()); 
	int *FN = (int*) malloc(sizeof(int)*Classes.size());

	memset(TP,0,sizeof(int)*Classes.size());
	memset(TN,0,sizeof(int)*Classes.size());
	memset(FP,0,sizeof(int)*Classes.size());
	memset(FN,0,sizeof(int)*Classes.size());

	for(i = 0; i < Classes.size(); i++){
		TP[i] = confusionMatrix[i][i];

		for(j = 0; j < Classes.size(); j++){
			if(j != i){
				FP[i] += confusionMatrix[j][i];
				FN[i] += confusionMatrix[i][j];
			}
		}
	}

	for(i = 0; i < Classes.size(); i++){
		for(j = 0; j < Classes.size(); j++){
			if(j != i){	
				for(int k = 0; k < Classes.size(); k++){
					if(k != i){
						TN[i] += confusionMatrix[j][k];
					}
				}
			}
		}
	}

	cout << "\n=== Detailed Accuracy By Class ===\n\n";

	XMLOutput << "\t\t<details>" << std::endl;

	int spacing = 10;

	cout << std::left << setw(spacing) << " "
		<< std::left << setw(spacing) << "Accuracy" 
		<< std::left << setw(spacing) << "TN Rate" 
		<< std::left << setw(spacing) << "FP Rate" 
		<< std::left << setw(spacing) << "Precision" 
		<< std::left << setw(spacing) << "Recall" 
		<< std::left << setw(spacing) << "F-Measure" 
		<< std::left << setw(spacing) << "Class" << endl;

	float Accuracy_mean = 0;
	float TNRate_mean = 0;
	float FPRate_mean = 0;
	float Precision_mean = 0;
	float Recall_mean = 0;
	float F1_Measure_mean = 0;


	for(i = 0; i < Classes.size() ; i++){

      float Accuracy = 0;
      float TNRate = 0; 
      float FPRate = 0; 
      float Precision = 0; 
      float Recall = 0;
      float FMeasure = 0;
		
		if(((float)(TP[i]+TN[i]+FN[i]+FP[i])) > 0) 
		   Accuracy = ((float)(TP[i]+TN[i]))/((float)(TP[i]+TN[i]+FN[i]+FP[i]));
		
		if(((float)(FP[i]+TN[i])) > 0)
		   TNRate = ((float)TN[i])/((float)(FP[i]+TN[i]));
		
		if(((float)(FP[i]+TN[i])) > 0)
		   FPRate = ((float)FP[i])/((float)(FP[i]+TN[i]));
		
		if(((float)(FP[i]+TP[i])) > 0)
		   Precision = ((float)TP[i])/((float)(FP[i]+TP[i]));
		
		if(((float)(FN[i]+TP[i])) > 0)
		   Recall = ((float)TP[i])/((float)(FN[i]+TP[i]));
		
      if((Recall+Precision) > 0)
         FMeasure = 2*Recall*Precision / (Recall+Precision);

		cout << std::left << setw(spacing) << " ";
		cout << std::left << setw(spacing) << Accuracy; //Accuracy
		cout << std::left << setw(spacing) << TNRate; //TN Rate
		cout << std::left << setw(spacing) << FPRate ; //FP Rate
		cout << std::left << setw(spacing) << Precision; //Precision
		cout << std::left << setw(spacing) << Recall; //Recall
		cout << std::left << setw(spacing) << FMeasure; //F1-Measure
		cout << std::left << setw(spacing) << ClassesLookup[i + 1] << endl; //Class


		XMLOutput << "\t\t\t<class name=\"" << ClassesLookup[i + 1] << "\" acc=\"" << Accuracy << "\" tnrate=\"" << TNRate << "\" fprate=\"" << FPRate << "\" precision=\"" << Precision << "\" recall=\"" << Recall << "\" fmeasure=\"" << FMeasure << "\"></class>" << std::endl;

      if(total_instances > 0){
	   	Accuracy_mean += Accuracy*((float)(TP[i]+FP[i])/total_instances);
   		TNRate_mean += TNRate*((float)(TP[i]+FP[i])/total_instances);
   		FPRate_mean += FPRate*((float)(TP[i]+FP[i])/total_instances);
   		Precision_mean += Precision*((float)(TP[i]+FP[i])/total_instances);
   		Recall_mean += Recall*((float)(TP[i]+FP[i])/total_instances);
   		F1_Measure_mean += (FMeasure*((float)(TP[i]+FP[i]))/total_instances);
	   }
	}


	cout << std::left << setw(spacing) << "W.Avg."; //Weighted Average
	cout << std::left << setw(spacing) << Accuracy_mean; //Accuracy
	cout << std::left << setw(spacing) << TNRate_mean; //TN Rate
	cout << std::left << setw(spacing) << FPRate_mean; //FP Rate
	cout << std::left << setw(spacing) << Precision_mean; //Precision
	cout << std::left << setw(spacing) << Recall_mean; //Recall
	cout << std::left << setw(spacing) << F1_Measure_mean; //F1-Measure
	cout << endl;

		
	XMLOutput << "\t\t\t<total acc=\"" << Accuracy_mean << "\" tnrate=\"" << TNRate_mean << "\" fprate=\"" << FPRate_mean << "\" precision=\"" << Precision_mean << "\" recall=\"" << Recall_mean << "\" fmeasure=\"" << F1_Measure_mean << "\"></total>" << std::endl;
	
	XMLOutput << "\t\t</details>" << std::endl;

	free(TP);
	free(TN);
	free(FP);
	free(FN);
	
	XMLOutput << "\t</output>\n";
	
}

void measures(int correct_instances,int total_instances, int incorrect_instances, int** confusionMatrix, std::map<string,int> Classes, std::map<int,string> ClassesLookup){

	int i,j;

	/*************SUMMARY************/
	cout << "\n=== Summary ===\n\n";
	cout << std::left << setw(50) << "Correctly Classified Instances " << std::left << setw(5) << correct_instances  << std::right << setw(5) << " " << ((float)correct_instances)/total_instances*100 << "%" << endl;
	cout << std::left << setw(50) << "Incorrectly Classified Instances " << std::left << setw(5) << incorrect_instances << std::right << setw(5) <<" " << ((float)incorrect_instances)/total_instances*100 << "%" <<  endl;
	cout << std::left << setw(50) << "Total Number of Instances " << std::left << setw(5) << total_instances << endl;
	
	/********CONFUSION MATRIX*********/
	cout << "\n=== Confusion Matrix ===\n\n";

	for(i = 0; i < Classes.size(); i++){
		cout << std::right << setw(5) << ((char) (i+97));
	}
	cout << "  <-- classified as " << endl;


	for(i = 0; i < Classes.size(); i++){
		for(j = 0; j < Classes.size(); j++){
			cout << std::right << setw(5) << confusionMatrix[i][j];
		}
		cout << std::right << setw(5) << " | " << ((char) (i+97)) << " = " << ClassesLookup[i+1] << endl;		
	}
	

	/********************************/

	int *TP = (int*) malloc(sizeof(int)*Classes.size());
	int *TN = (int*) malloc(sizeof(int)*Classes.size());
	int *FP = (int*) malloc(sizeof(int)*Classes.size()); 
	int *FN = (int*) malloc(sizeof(int)*Classes.size());

	memset(TP,0,sizeof(int)*Classes.size());
	memset(TN,0,sizeof(int)*Classes.size());
	memset(FP,0,sizeof(int)*Classes.size());
	memset(FN,0,sizeof(int)*Classes.size());

	for(i = 0; i < Classes.size(); i++){
		TP[i] = confusionMatrix[i][i];

		for(j = 0; j < Classes.size(); j++){
			if(j != i){
				FP[i] += confusionMatrix[j][i];
				FN[i] += confusionMatrix[i][j];
			}
		}
	}

	for(i = 0; i < Classes.size(); i++){
		for(j = 0; j < Classes.size(); j++){
			if(j != i){	
				for(int k = 0; k < Classes.size(); k++){
					if(k != i){
						TN[i] += confusionMatrix[j][k];
					}
				}
			}
		}
	}

	cout << "\n=== Detailed Accuracy By Class ===\n\n";

	int spacing = 10;

	cout << std::left << setw(spacing) << " "
		<< std::left << setw(spacing) << "Accuracy" 
		<< std::left << setw(spacing) << "TN Rate" 
		<< std::left << setw(spacing) << "FP Rate" 
		<< std::left << setw(spacing) << "Precision" 
		<< std::left << setw(spacing) << "Recall" 
		<< std::left << setw(spacing) << "F-Measure" 
		<< std::left << setw(spacing) << "Class" << endl;

	float Accuracy_mean = 0;
	float TNRate_mean = 0;
	float FPRate_mean = 0;
	float Precision_mean = 0;
	float Recall_mean = 0;
	float F1_Measure_mean = 0;


	for(i = 0; i < Classes.size() ; i++){

      float Accuracy = 0;
      float TNRate = 0; 
      float FPRate = 0; 
      float Precision = 0; 
      float Recall = 0;
      float FMeasure = 0;
		
		if(((float)(TP[i]+TN[i]+FN[i]+FP[i])) > 0) 
		   Accuracy = ((float)(TP[i]+TN[i]))/((float)(TP[i]+TN[i]+FN[i]+FP[i]));
		
		if(((float)(FP[i]+TN[i])) > 0)
		   TNRate = ((float)TN[i])/((float)(FP[i]+TN[i]));
		
		if(((float)(FP[i]+TN[i])) > 0)
		   FPRate = ((float)FP[i])/((float)(FP[i]+TN[i]));
		
		if(((float)(FP[i]+TP[i])) > 0)
		   Precision = ((float)TP[i])/((float)(FP[i]+TP[i]));
		
		if(((float)(FN[i]+TP[i])) > 0)
		   Recall = ((float)TP[i])/((float)(FN[i]+TP[i]));
		
      if((Recall+Precision) > 0)
         FMeasure = 2*Recall*Precision / (Recall+Precision);

		cout << std::left << setw(spacing) << " ";
		cout << std::left << setw(spacing) << Accuracy; //Accuracy
		cout << std::left << setw(spacing) << TNRate; //TN Rate
		cout << std::left << setw(spacing) << FPRate ; //FP Rate
		cout << std::left << setw(spacing) << Precision; //Precision
		cout << std::left << setw(spacing) << Recall; //Recall
		cout << std::left << setw(spacing) << FMeasure; //F1-Measure
		cout << std::left << setw(spacing) << ClassesLookup[i + 1] << endl; //Class


      if(total_instances > 0){
	   	Accuracy_mean += Accuracy*((float)(TP[i]+FP[i])/total_instances);
   		TNRate_mean += TNRate*((float)(TP[i]+FP[i])/total_instances);
   		FPRate_mean += FPRate*((float)(TP[i]+FP[i])/total_instances);
   		Precision_mean += Precision*((float)(TP[i]+FP[i])/total_instances);
   		Recall_mean += Recall*((float)(TP[i]+FP[i])/total_instances);
   		F1_Measure_mean += (FMeasure*((float)(TP[i]+FP[i]))/total_instances);
	   }
	}


	cout << std::left << setw(spacing) << "W.Avg."; //Weighted Average
	cout << std::left << setw(spacing) << Accuracy_mean; //Accuracy
	cout << std::left << setw(spacing) << TNRate_mean; //TN Rate
	cout << std::left << setw(spacing) << FPRate_mean; //FP Rate
	cout << std::left << setw(spacing) << Precision_mean; //Precision
	cout << std::left << setw(spacing) << Recall_mean; //Recall
	cout << std::left << setw(spacing) << F1_Measure_mean; //F1-Measure
	cout << endl;

	free(TP);
	free(TN);
	free(FP);
	free(FN);

	
}