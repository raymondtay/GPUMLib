/*
	Noel Lopes is a Professor Assistant at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009 Noel de Jesus Mendonça Lopes

	This file is part of Multiple Back-Propagation.

    Multiple Back-Propagation is free software: you can redistribute it and/or modify
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

/**
 Class    : Pointer<Type>
 Puropse  : Create a pointer of any type that automaticaly destroys the
            object that is pointing to when there are no more pointers to
            that object.
 Date     : 4 of July of 1999
 Reviewed : 17 of November of 1999
 Version  : 1.1.0
 Comments : Each Pointer is whithin a list of Pointer's that point to the
            same object.
*/
#ifndef Pointer_h
#define Pointer_h

#include <assert.h>

template <class Type> class Pointer {
	private :
		/**
		 Attribute : Pointer<Type> * previous
		 Purpose   : pointer to the previous Pointer that points to the same
		             object as this Pointer.
		*/
		Pointer<Type> * previous;

		/**
		 Attribute : Pointer<Type> * next
		 Purpose   : pointer to the next Pointer that points to the same object
		             as this Pointer.
		*/
		Pointer<Type> * next;

		/**
		 Attribute : Type * object
		 Purpose   : object to where this Pointer is points to.
		*/
		Type * object;

		/**
		 Method   : void Destroy()
		 Purpose  : Destroy the Pointer.
		 Version  : 1.0.1
		 Comments : Which means that this Pointer will stop pointing at any
		            object. This also means that 1) the Pointer will be removed
		            from the list where it was inserted and 2) if there where no
		            other Pointers to the object where this Pointer was pointing,
		            the object will be deleted.
		*/
		void Destroy() {
			if (previous == NULL && next == NULL) {
				delete object;
			} else {
				if (previous != NULL) previous->next = next;
				if (next != NULL) next->previous = previous;
				previous = next = NULL;
			}

			object = NULL;
		}

		/**
		 Method   : void Connect(Pointer<Type> * other)
		 Purpose  : Connect this Pointer to other Pointer.
		 Version  : 1.0.2
		 Comments : Which means that this Pointer will point to the same object
		            as the other pointer, consequently it will be inserted in the
		            list where the other Pointer belongs.
		*/
		void Connect(Pointer<Type> * other) {
			if (object == other->object) return;

			Destroy();

			if (other->object != NULL) {
				while (other->next != NULL) other = other->next;
				other->next = this;

				object = other->object;
				previous = other;
			}
		}
	
	public :
		/**
		 Constructor : Pointer(Type * newObject = NULL)
		 Purpose     : Create a Pointer object that will point to a newly created
		               object.
		 Version     : 1.1.0
		*/
		Pointer(Type * newObject = NULL) {
			object = newObject;
			previous = next = NULL;
		}

		/**
		 Constructor : Pointer(Pointer<Type> & other)
		 Purpose     : Create a Pointer object that will point to an existing 
		               Pointer.
		 Version     : 1.1.1
		*/
		Pointer(Pointer<Type> & other) {
			previous = next = NULL;
			object = NULL;
			Connect(&other);
		}

		/**
		 Destructor : ~Pointer()
		 Purpose    : Destroy the Pointer.
		 Version    : 1.0.0
		*/
		~Pointer() {
			Destroy();
		}

		/**
		 Operator : Pointer<Type> & operator = (Pointer<Type> other)
		 Purpose  : Assign other Pointer to this Pointer.
		 Version  : 1.0.0
		*/
		Pointer<Type> & operator = (Pointer<Type> & other) {
			Connect(&other);
			return *this;
		}

		/**
		 Operator : Pointer<Type> & operator = (Type * newObject)
		 Purpose  : Assign a newly created object to this Pointer.
		 Version  : 1.0.0
		*/
		Pointer<Type> & operator = (Type * newObject) {
			Destroy();
			object = newObject;
			return *this;
		}

		/**
		 Operator : Type * operator ->() const
		 Purpose  : Allows to access directly the members of the object to where
		            this pointer points to.
		 Version  : 1.0.0
		*/
		Type * operator ->() const {
			assert (object != NULL);
			return object;
		}

		/**
		 Operator : operator Type * () const
		 Purpose  : Returns a pointer to the object where this Pointer is
		            pointing to.
		 Version  : 1.0.0
		*/
		operator Type * () const {
			return object;
		}

		/**
		 Method   : bool IsNull() const
		 Purpose  : Returns whether this Pointer is pointing to an object or not.
		 Version  : 1.0.0
		*/
		bool IsNull() const {
			return (object == NULL);
		}
};

#endif