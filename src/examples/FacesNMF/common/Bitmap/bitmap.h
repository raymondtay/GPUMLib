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
 Class    : Bitmap
 Purpose  : Bitmap class.
 Date     : 15 of January of 2000.
 Reviewed : 11 of February of 2000.
 Version  : 1.0.0
 Comments : 
             ---------
            | CObject |
             --------- 
                |   ------------
                -->| CGdiObject |
                    ------------
                      |   ---------
                      -->| CBitmap |
                          ---------
                            |   --------
                            -->| Bitmap |
                                --------
*/
#ifndef Bitmap_h
#define Bitmap_h

class Bitmap : public CBitmap {
	private :
		/**
		 Attribute : LONG height
		 Purpose   : Contains the bitmap height.
		*/
		LONG height;

		/**
		 Attribute : LONG width
		 Purpose   : Contains the bitmap width.
		*/
		LONG width;

		/**
		 Method  : void DetermineWidthHeight()
		 Purpose : Determine the width and the height of the bitmap.
		 Version : 1.0.0
		*/
		void DetermineWidthHeight() {
			BITMAP bitmapInfo;

			if (GetBitmap(&bitmapInfo)) {
				width  = bitmapInfo.bmWidth;
				height = bitmapInfo.bmHeight;
			}
		}

	public :
		/**
		 Constructor : Bitmap()
		 Purpose     : Create a bitmap.
		 Version     : 1.0.0
		*/
		Bitmap() {
			height = width = -1;
		}

		/**
		 Method  : LONG Height()
		 Purpose : Returns the bitmap height.
		 Version : 1.0.0
		*/
		LONG Height() {
			if (height == -1) DetermineWidthHeight();
			return height;
		}

		/**
		 Method  : LONG Width()
		 Purpose : Returns the bitmap width.
		 Version : 1.0.0
		*/
		LONG Width() {
			if (width == -1) DetermineWidthHeight();
			return width;
		}

		/**
		 Method  : CBitmap * ShrinkBitmap(LONG bx, LONG by)
		 Purpose : Returns this bitmap shrinked.
		 Version : 1.0.0
		*/
		CBitmap * ShrinkBitmap(LONG bx, LONG by);
};

#endif