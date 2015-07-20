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
 Class    : FlickerFreeDC
 Purpose  : Flicker free device context.
 Date     : 14 of January of 2000
 Reviewed : 16 of May of 2008
 Version  : 1.4.1
 Comments : parts of this class were based on Keith Rule class

             ---------
            | CObject |
             --------- 
                |   -----
                -->| CDC |
                    -----
                      |   ---------------
                      -->| FlickerFreeDC |
                          ---------------
*/
#ifndef FlickerFreeDC_h
#define FlickerFreeDC_h

#include <afxctl.h>
#include "../Bitmap/Bitmap.h"
#include "../Pointers/Pointer.h"

class FlickerFreeDC : public CDC {
	private:
		/**
		 Attribute : CBitmap bitmap
		 Purpose   : Bitmap of the Memory Device Context.
		 Comments  : GDI output functions can be used with a memory 
		             device context only if a bitmap has been created 
								 and selected into that context.
		*/
		CBitmap bitmap;		
		
		/**
		 Attribute : CBitmap * oldBitmap
		 Purpose   : Keep the initial bitmap of the Memory Device Context.
		*/
		CBitmap * oldBitmap;

		/**
		 Attribute : CDC * dc
		 Purpose   : Pointer to the Device Context where we are drawing.
		*/
		CDC * dc;
		
		/**
		 Attribute : CRect invalidRect
		 Purpose   : Rectangle that needs to be draw.
		*/
		CRect invalidRect;

		/**
		 Attribute : CFont * previousFont
		 Purpose   : Pointer to the previous font selected.
		*/
		CFont * previousFont;

		/**
		 Attribute : Pointer<CPaintDC> paintDC
		 Purpose   : Pointer to the Paint Device Context.
		 Comments  : Used when no device context has been created for a window.
		*/
		Pointer<CPaintDC> paintDC;

		/**
		 Method  : void Initialize(CDC * dc, const CRect & invalidRect, COLORREF backColor)
		 Purpose : Initialize the flicker free device context.
		 Version : 1.1.0
		*/
		void Initialize(CDC * dc, const CRect & invalidRect, COLORREF backColor) {
			if (!CreateCompatibleDC(this->dc = dc) || !bitmap.CreateCompatibleBitmap(dc, invalidRect.Width(), invalidRect.Height())) {
				Attach(dc->GetSafeHdc());
			} else {
				this->invalidRect = invalidRect;
				oldBitmap = SelectObject(&bitmap);
				SetWindowOrg(invalidRect.TopLeft());
			}

			FillSolidRect(invalidRect, backColor);

			previousFont = NULL;
		}

		/**
		 Method  : void Initialize(CWnd * window, COLORREF backColor)
		 Purpose : Initialize the flicker free device context.
		 Version : 1.0.0
		*/
		void Initialize(CWnd * window, COLORREF backColor) {
			window->GetUpdateRect(&invalidRect);
			paintDC = new CPaintDC(window);
			Initialize(paintDC, invalidRect, backColor);
		}

	public:
		/**
		 Constructor : FlickerFreeDC(CWnd * window, COLORREF backColor)
		 Purpose     : Create a flicker free device context for a given window.
		 Version     : 1.0.0
		*/
		FlickerFreeDC(CWnd * window, COLORREF backColor) {
			Initialize(window, backColor);
		}

		/**
		 Constructor : FlickerFreeDC(CWnd * window)
		 Purpose     : Create a flicker free device context for a given window.
		 Version     : 1.0.0
		*/
		FlickerFreeDC(CWnd * window) {
			Initialize(window, GetSysColor(COLOR_WINDOW));
		}

		/**
		 Constructor : FlickerFreeDC(CDC * dc, const CRect & invalidRect, COLORREF backColor)
		 Purpose     : Create a flicker free device context.
		 Version     : 1.1.0
		*/
		FlickerFreeDC(CDC * dc, const CRect & invalidRect, COLORREF backColor) {
			Initialize(dc, invalidRect, backColor);
		}

		/**
		 Constructor : FlickerFreeDC(CDC * dc, const CRect & invalidRect)
		 Purpose     : Create a flicker free device context with the same
		               back color as the default window back color.
		 Version     : 1.1.0
		*/
		FlickerFreeDC(CDC * dc, const CRect & invalidRect) {
			Initialize(dc, invalidRect, GetSysColor(COLOR_WINDOW));
		}

		/**
		 Destructor : ~FlickerFreeDC()
		 Purpose    : If the Memory Device context and his bitmap where 
		              sucessfully created draw the his bitmap on the 
									device context, making the image flicker free.
		 Version    : 1.1.0
		*/
		virtual ~FlickerFreeDC() {
			if (previousFont != NULL) SelectObject(previousFont);

			if (GetSafeHdc() != dc->GetSafeHdc()) {
				dc->BitBlt(invalidRect.left, invalidRect.top, invalidRect.Width(), invalidRect.Height(), this, invalidRect.left, invalidRect.top, SRCCOPY);
				SelectObject(oldBitmap);
			} else {
				Detach();
			}
		}

		/**
		 Method  : void DrawBitmap(UINT nIDResource, int x, int y, DWORD dwRop = SRCCOPY)
		 Purpose : Draw a bitmap contained in the specified resource.
		 Version : 1.0.1
		*/
		void DrawBitmap(UINT nIDResource, int x, int y, DWORD dwRop = SRCCOPY) {
			CBitmap b;
			if (b.LoadBitmap(nIDResource)) {
				CDC memoryDC;
				if (memoryDC.CreateCompatibleDC(this)) {
					BITMAP bitmapInfo;
					if (b.GetBitmap(&bitmapInfo)) {
						memoryDC.SelectObject(&b);
						BitBlt(x, y, bitmapInfo.bmWidth, bitmapInfo.bmHeight, &memoryDC, 0, 0, dwRop);
					}
				}
			}
		}

		/**
		 Method   : void DrawBitmap(Bitmap * b, int x, int y, int width, int height, DWORD dwRop = SRCCOPY)
		 Purpose  : Stretch and Draw a bitmap.
		 Version  : 1.0.0
		*/
		/*void DrawBitmap(Bitmap * b, int x, int y, int width, int height, DWORD dwRop = SRCCOPY) {
			CDC memoryDC;

			if (memoryDC.CreateCompatibleDC(this)) {				
				memoryDC.SelectObject(b);
				StretchBlt(x, y, width, height, &memoryDC, 0, 0, b->Width(), b->Height(), dwRop);
			}
		}*/

		/**
		 Method   : void DrawBitmap(Bitmap * b, int x, int y, int width, int height, DWORD dwRop = SRCCOPY)
		 Purpose  : Stretch and Draw a bitmap.
		 Version  : 1.0.0
		*/
		void DrawBitmap(Bitmap * b, int x, int y, DWORD dwRop = SRCCOPY) {
			CDC memoryDC;

			if (memoryDC.CreateCompatibleDC(this)) {				
				memoryDC.SelectObject(b);
				BitBlt(x, y, b->Width(), b->Height(), &memoryDC, 0, 0, dwRop);
			}
		}

		/**
		 Method   : void DrawBitmap(Bitmap * b, int x, int y, int width, int height, DWORD dwRop = SRCCOPY)
		 Purpose  : Stretch and Draw a bitmap.
		 Version  : 1.0.0
		*/
		void DrawBitmap(CBitmap * b, int x, int y, DWORD dwRop = SRCCOPY) {
			CDC memoryDC;

			int width  = 0;
			int height = 0;

			if (memoryDC.CreateCompatibleDC(this)) {
				memoryDC.SelectObject(b);

				BITMAP bitmapInfo;

				if (b->GetBitmap(&bitmapInfo)) {
					width  = bitmapInfo.bmWidth;
					height = bitmapInfo.bmHeight;
				}

				BitBlt(x, y, width, height, &memoryDC, 0, 0, dwRop);
			}
		}
		
		/**
		 Method  : void Line(int xi, int yi, int xf, int yf)
		 Purpose : Draw a line.
		 Version : 1.0.0
		*/
		void Line(int xi, int yi, int xf, int yf) {
			MoveTo(xi, yi);
			LineTo(xf, yf);
		}

		/**
		 Method  : void Line(POINT a, POINT b)
		 Purpose : Draw a line.
		 Version : 1.0.0
		*/
		void Line(POINT a, POINT b) {
			MoveTo(a);
			LineTo(b);
		}

		/**
		 Method  : void Triangle(POINT a, POINT b, POINT c)
		 Purpose : Draw a triangle.
		 Version : 1.0.0
		*/
		void Triangle(POINT a, POINT b, POINT c) {
			POINT points[3];

			points[0] = a;
			points[1] = b;
			points[2] = c;

			Polygon(points, 3);
		}

		/**
		 Method  : void HorizontalArrow(int y, int xi, int xf)
		 Purpose : Draw an arrow from left to right.
		 Version : 1.0.0
		*/
		void HorizontalArrow(int y, int xi, int xf) {
			if (xi < xf) {				
				Line(xi, y, xf + 1, y);
				Line(xf-1, y-1, xf-1 , y+2);
				Line(xf-2, y-2, xf-2 , y+3);
				Line(xf-3, y-3, xf-3 , y+4);
			} else {
				Line(xf, y, xi + 1, y);
				Line(xf+1, y-1, xf+1 , y+2);
				Line(xf+2, y-2, xf+2 , y+3);
				Line(xf+3, y-3, xf+3 , y+4);
			}
		}

		/**
		 Method  : VerticalArrow(int x, int yi, int yf)
		 Purpose : Draw an arrow from left to right.
		 Version : 1.0.0
		*/
		void VerticalArrow(int x, int yi, int yf) {
			if (yi < yf) {
				Line(x, yi, x, yf + 1);
				Line(x-1, yf-1, x+2 , yf-1);
				Line(x-2, yf-2, x+3 , yf-2);
				Line(x-3, yf-3, x+4 , yf-3);
			} else {
				Line(x, yf, x, yi + 1);
				Line(x-1, yf+1, x+2 , yf+1);
				Line(x-2, yf+2, x+3 , yf+2);
				Line(x-3, yf+3, x+4 , yf+3);
			}
		}

		/**
		 Method  : void SelectFont(CFontHolder & f)
		 Purpose : Select a font.
		 Version : 1.0.0
		*/
		void SelectFont(CFontHolder & f) {
			if (previousFont != NULL) SelectObject(previousFont);
			previousFont = SelectObject(CFont::FromHandle(f.GetFontHandle()));
		}

		/**
		 Method  : void SelectFont(CFont * f)
		 Purpose : Select a font.
		 Version : 1.0.0
		*/
		void SelectFont(CFont * f) {
			if (previousFont != NULL) SelectObject(previousFont);
			previousFont = SelectObject(f);
		}
};

#endif