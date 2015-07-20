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

#include "stdafx.h"
#include <assert.h>
#include "Bitmap.h"
#include "../Pointers/Matrix.h"

#define RGB_BYTES 3
#define RGB_BITS 24

CBitmap * Bitmap::ShrinkBitmap(LONG width, LONG height) {
	LONG originalWidth = Width();
	LONG originalHeight = Height();

	assert (width <= originalWidth && height <= originalWidth);

	HDC hdc = GetDC(NULL);

	BITMAPINFO bitmapInfo;
	bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfo.bmiHeader.biPlanes = 1;
	bitmapInfo.bmiHeader.biBitCount = RGB_BITS;
	bitmapInfo.bmiHeader.biCompression = BI_RGB;
	bitmapInfo.bmiHeader.biSizeImage = 0;
	bitmapInfo.bmiHeader.biXPelsPerMeter = 0;
	bitmapInfo.bmiHeader.biYPelsPerMeter = 0;
	bitmapInfo.bmiHeader.biClrUsed = 0;
	bitmapInfo.bmiHeader.biClrImportant = 0;

	// Obtain the bitmap bytes
	LONG bytesPerRow = (originalWidth + 1) * RGB_BYTES & 0xFFFC; // (32-bit alignment)
	Matrix<BYTE> bitmapBytes(originalHeight, bytesPerRow);
	bitmapInfo.bmiHeader.biWidth = originalWidth;
	bitmapInfo.bmiHeader.biHeight = originalHeight;
    GetDIBits(hdc, (HBITMAP) *this, 0, originalHeight, bitmapBytes.Pointer(), &bitmapInfo, DIB_RGB_COLORS);

	// Create the array that will have the shrinked bitmap bytes
	LONG shrinkedBytesPerRow = width * RGB_BYTES;
	LONG shrinkedBytesPerRow32bitAligned = (shrinkedBytesPerRow + RGB_BYTES) & 0xFFFC; // (32-bit alignment)
	Matrix<BYTE> shrinkedBitmapBytes(height, shrinkedBytesPerRow32bitAligned);

	// Create an array that will have the sum of the original bitmap bytes that compose a shrinked bitmap byte
	Matrix<float> sumOriginalBitmapBytes(height, shrinkedBytesPerRow);
	float * sum = sumOriginalBitmapBytes.Pointer();
	DWORD shrinkCol = 0;

	// fill the array that will have the sum of the original bitmap bytes that compose a shrinked bitmap byte
	LONG heightCompleted = 0;
	for(LONG r = 0; r < originalHeight; r++) {
		BYTE * col = bitmapBytes[r].Pointer();	
		LONG widthCompleted = 0;

		heightCompleted += height;

		if (heightCompleted <= originalHeight) {
			for(LONG c = 0; c < originalWidth; c++) {
				widthCompleted += width;

				if (widthCompleted <= originalWidth) {
					for(int b=0; b<RGB_BYTES; b++) sum[shrinkCol + b] += *col++;

					if (widthCompleted == originalWidth) {
						shrinkCol += RGB_BYTES;
						widthCompleted = 0;
					}
				} else {
					float percentFirstByte = (widthCompleted - originalWidth) / (float) width;
					float percentLastByte  = 1 - percentFirstByte;

					for(int b=0; b<RGB_BYTES; b++) {
						sum[shrinkCol] += *col * percentLastByte;
						sum[shrinkCol + RGB_BYTES] += *col * percentFirstByte;
						shrinkCol++;
						col++;
					}

					widthCompleted -= originalWidth;
				}
			}

			if (heightCompleted < originalHeight) {
				shrinkCol -= shrinkedBytesPerRow;
			} else {
				heightCompleted = 0;
			}
		} else {
			float heightPercentFirstByte = (heightCompleted - originalHeight) / (float) height;
			float heightPercentLastByte  = 1 - heightPercentFirstByte;

			for(LONG c = 0; c < originalWidth; c++) {
				widthCompleted += width;

				if (widthCompleted <= originalWidth) {
					for(int b=0; b<RGB_BYTES; b++) {
						sum[shrinkCol + b] += *col * heightPercentLastByte;
						sum[shrinkCol + shrinkedBytesPerRow + b] += *col * heightPercentFirstByte;
						col++;
					}

					if (widthCompleted == originalWidth) {
						shrinkCol += RGB_BYTES;
						widthCompleted = 0;
					}
				} else {
					float widthPercentFirstByte = (widthCompleted - originalWidth) / (float) width;
					float widthPercentLastByte  = 1 - widthPercentFirstByte;

					for(int b=0; b<RGB_BYTES; b++) {
						sum[shrinkCol] += *col * heightPercentLastByte * widthPercentLastByte;
						sum[shrinkCol + RGB_BYTES] += *col * heightPercentLastByte * widthPercentFirstByte;
						sum[shrinkCol + shrinkedBytesPerRow] += *col * heightPercentFirstByte * widthPercentLastByte;
						sum[shrinkCol + shrinkedBytesPerRow + RGB_BYTES] = *col * heightPercentFirstByte * widthPercentFirstByte;
						shrinkCol++;
						col++;
					}

					widthCompleted -= originalWidth;
				}
			}

			heightCompleted -= originalHeight;
		}
	}

	// fill the array that will have the shrinked bitmap bytes
	float factor = originalWidth * originalHeight / (float) (width * height);
	sum = sumOriginalBitmapBytes.Pointer(); 
	for(LONG r = 0; r < height; r++) {	
		Array<BYTE> row = shrinkedBitmapBytes[r];
		for(LONG rgbCol = 0; rgbCol < shrinkedBytesPerRow; rgbCol++) row[rgbCol] = (BYTE) (*sum++ / factor + 0.5);
	}

	// Create the shrinked bitmap
	bitmapInfo.bmiHeader.biWidth = width;
	bitmapInfo.bmiHeader.biHeight = height;
	HBITMAP shrinkedBitmap = ::CreateCompatibleBitmap(hdc, width, height);
	SetDIBits (hdc, shrinkedBitmap, 0, height, shrinkedBitmapBytes.Pointer(), &bitmapInfo, DIB_RGB_COLORS);

	ReleaseDC (NULL,hdc);

	return CBitmap::FromHandle(shrinkedBitmap);
}