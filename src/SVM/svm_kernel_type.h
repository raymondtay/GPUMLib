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

#ifndef SVM_KERNEL_TYPE_H_
#define SVM_KERNEL_TYPE_H_

namespace GPUMLib {

	//! Type of kernel function
	enum svm_kernel_type {
		SVM_KT_LINEAR, SVM_KT_POLYNOMIAL, SVM_KT_RBF, SVM_KT_SIGMOID, SVM_KT_UKF
	};

}
#endif
