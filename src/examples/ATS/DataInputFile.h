/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011 Noel de Jesus Mendonça Lopes

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

#ifndef DataInputFile_h
#define DataInputFile_h

#include "../../common/CudaDefinitions.h"

#include <sstream>
#include <fstream>
#include <limits>

using namespace std;

class DataInputFile {
	private:
		ifstream f;
		bool csvfile;
	
	public:
		DataInputFile(string & filename) {
			f.open(filename.c_str());
			
			csvfile = false;
			size_t pos = filename.length() - 4;
			if (pos > 0) {
				if (filename[pos] == '.' && (filename[pos + 1] == 'c' || filename[pos + 1] == 'C') && (filename[pos + 2] == 's' || filename[pos + 2] == 'S') && (filename[pos + 3] == 'v' || filename[pos + 2] == 'V')) {
					csvfile = true;
				}
			}
		}

		ifstream & GetStream() {
			return f;
		}

		cudafloat GetNextValue(bool lastvalue) {
			cudafloat v;

			if (!csvfile) {
				f >> v;
			} else {
				string line;

				if (lastvalue) {
					getline(f, line);
				} else {
					getline(f, line, ',');
				}

				size_t length = line.length();

				size_t p;
				for(p = 0; p < length; p++) {
					if (!isspace(line[p])) break;
				}

				if (p == length || line[p] == '?') {
					v = numeric_limits<cudafloat>::quiet_NaN();
				} else {
					stringstream ss(line);
					ss >> v;
				}
			}

			return v;
		}

		bool eof() {
			return f.eof();
		}

		void Close() {
			f.close();
		}

		void IgnoreLine() {
			string line;
			getline(f, line);
		}
};

#endif
