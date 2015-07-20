/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011, 2012 Noel de Jesus Mendonça Lopes

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

#ifndef ConfusionMatrix_h
#define ConfusionMatrix_h

#include <iostream>
#include "../../memory/HostMatrix.h"

using namespace std;
using namespace GPUMLib;

class ConfusionMatrix {
	private:
		HostMatrix<int> results;
		int classes;

		int TP(int _class) const {
			return results(_class, _class);
		}

		int FP(int _class) const {
			int fp = 0;
			for(int c = 0; c < classes; c++) if (c != _class) fp += results(c, _class);

			return fp;
		}

		int FN(int _class) const {
			int fn = 0;
			for(int c = 0; c < classes; c++) if (c != _class) fn += results(_class, c);

			return fn;
		}

		int Positives(int _class) const {
			return TP(_class) + FN(_class); 
		}

		double Precision(int _class) const {
			int tp = TP(_class);

			if (tp == 0) {
				return (Positives(_class) == 0) ? 1.0 : 0.0;
			} else {
				return (double) tp / (tp + FP(_class));
			}
		}

		double Recall(int _class) const {
			int positives = Positives(_class);

			if (positives == 0) {
				return 1.0;
			} else {
				return (double) TP(_class) / positives;
			}
		}

	public:
		ConfusionMatrix(int classes) {
			this->classes = classes;
			results.ResizeWithoutPreservingData(classes, classes);

			Reset();
		}

		void Reset() {
			for (int c = 0; c < classes; c++) {
				for (int p = 0; p < classes; p++) results(c, p) = 0;
			}
		}

		void Classify(int correctClass, int predictedClass) {
			results(correctClass, predictedClass)++;
		}

		double Precision() const {
			//if (classes == 2) return Precision(1);

			int count = 0;
			double sum = 0.0;
			for(int c = 0; c < classes; c++) {
				if (Positives(c) > 0) {
					sum += Precision(c);
					count++;
				}
			}

			return sum / count;
		}

		double Recall() const {
			//if (classes == 2) return Recall(1);

			int count = 0;
			double sum = 0.0;
			for(int c = 0; c < classes; c++) {
				if (Positives(c) > 0) {
					sum += Recall(c);
					count++;
				}
			}

			return sum / count;
		}

		double FMeasure() const {
			double precision = Precision();
			if (precision == 0) return 0.0;

			double recall = Recall();
			return  2.0 * precision * recall / (precision + recall);
		}

		double Accuracy() const {
			double correct = 0;
			double total = 0;

			for(int c = 0; c < classes; c++) {
				for(int p = 0; p < classes; p++) {
					int classified = results(c, p);

					if (c == p) correct += classified;
					total += classified;
				}
			}

			return correct / total;
		}

		void Show() const {
			cout << endl << "\t\tPredicted" << endl;
			cout << "actual\t";

			for(int p = 0; p < classes; p++) cout << '\t' << p;
			cout << endl << endl;

			for(int c = 0; c < classes; c++) {
				cout << c << '\t';
				for(int p = 0; p < classes; p++) cout << '\t' << results(c, p);
				cout << endl;
			}
			cout << endl;
		}
};

#endif