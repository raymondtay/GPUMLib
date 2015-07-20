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

#ifndef OutputFile_h
#define OutputFile_h

#define PRECISION 15

#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;

class OutputFile {
	private:
		ofstream f;
		
	public:
		OutputFile(const char * filename) : f(filename, ios::out) {}

		void Write(const char * s) {
			f << s;
		}

		void Write(int v) {
			ostringstream sstream;
	
			sstream << v;
			
			Write(sstream.str().c_str());
		}

		void Write(double v) {
			ostringstream sstream;
	
			sstream << fixed << setprecision (PRECISION) << v;
			
			Write(sstream.str().c_str());
		}

		void WriteLine() {
			f << endl;
		}

		void WriteLine(const char * s) {
			f << s << endl;
		}

		void WriteLine(int v) {
			Write(v);
			f << endl;
		}

		void WriteLine(double v) {
			Write(v);
			f << endl;
		}
		
		void WriteLine(unsigned v) {
			ostringstream sstream;	
			sstream << v;			
			WriteLine(sstream.str().c_str());
		}

		void WriteLine(bool value) {
			f << ((value) ? "1" : "0") << endl;
		}
};

#endif