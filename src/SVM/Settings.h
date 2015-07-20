/*
Joao Goncalves is a MSc Student at the University of Coimbra, Portugal
Copyright (C) 2012 Joao Goncalves

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

/*
* Settings.h
*
*  Created on: Jan 5, 2012
*      Author: Joao Carlos Ferreira Goncalves
*/

#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <vector>

namespace GPUMLib {

	//! This class acts like a placeholder for an argument, composed of a parameter and a attribute/value
	class Argument {

	public:
		//! An parameter/argument
		char* argument;
		//! A value associated with a given argument/parameter
		char* value; 	// this could be updated (in the future) to have multiple values

		//! Creates an empty argument, without parameters
		Argument() {
			argument = NULL;
			value = NULL;
		}
	};

	//! Utility class to parse main()'s arguments, store them in a convenient list and access when needed
	class Settings {

	private:
		vector<Argument*> *argument_list;
		int argc;
		char **argv;

		//! Parses the given argc and argv and creates a list with the arguments
		void createSettings() {
			//first argument is the executable, so ignore it
			int argc_m1 = argc - 1;
			for (int i = 1; i < argc; i++) {
				//	cout << argv[i] << endl;
				//arguments must be on the form
				//-arg <val>
				//where val can be non-existent
				char* cur_arg = argv[i];
				//if it is an argument, it begins with a "-"
				if (cur_arg[0] == '-') {
					Argument *a = new Argument();
					a->argument = cur_arg;
					argument_list->push_back(a);
					//check if next string is the value
					if (i < argc_m1) { //obviously if there are more arguments
						char* next_arg = argv[i + 1];
						if (next_arg[0] != '-') {
							a->value = next_arg;
							i++;
						}
					}
				}
				//		cout << argument_list->size() << endl;
			}
		}

	public:
		//! Receives the same arguments as the main function and creates an internal list with the arguments
		//! \param argc The same argc as given in main()
		//! \param argv The same argv as given in main()
		Settings(int argc, char **argv) {
			argument_list = new vector<Argument*>();
			this->argc = argc;
			this->argv = argv;
			this->createSettings();
		}

		//! Call this to free the internal structures created after using the constructor
		~Settings() {
			for (size_t i = 0; i < argument_list->size(); i++) {
				delete argument_list->at(i);
			}
			delete argument_list;
		}

		//! Gets the argument at the given position (the nth argument)
		//! \param pos The position of the argument to be returned
		//! \return The Argument at position pos
		Argument* getArgument(size_t pos) {
			return this->argument_list->at(pos);
		}

		//! Gets the number of parsed arguments
		//! \return The number of parsed arguments
		size_t getAmountArguments() {
			return this->argument_list->size();
		}
	};

} // namespace GPUMLib

#endif /* SETTINGS_H_ */
