/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal
	Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014 Noel de Jesus Mendonça Lopes

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

// FacesNMFDlg.h : header file
//

#pragma once
#include "../common/CudaInit.h"
#include "..\..\NMF\NMFmultiplicativeDivergence.h"
#include "..\..\NMF\NMFmultiplicativeEuclidian.h"
#include "..\..\NMF\NMFadditiveEuclidian.h"
#include "..\..\NMF\NMFAdditiveDivergence.h"
#include "common/FlickerFreeDC/FlickerFreeDC.h"
#include "HostNMF.h"

#include "afxcmn.h"
#include "afxwin.h"

// CFacesNMFDlg dialog
class CFacesNMFDlg : public CDialog
{
private:
	NMF_METHOD nmfMethod;

	HostNMF * hnmf;
	HostNMF * hnmfTest;

	NMF * nmf;
	bool nmfContainsTrainData;

	Matrix<cudafloat> Vtest;
	Matrix<cudafloat> Wrescaled;

	int numberImages;
	int numberImagesTest;
	int imageWidth;
	int imageHeight;

	int rank;

// Construction
public:
	CFacesNMFDlg(CWnd* pParent = NULL);	// standard constructor

	~CFacesNMFDlg() {
		if (hnmf != NULL) delete hnmf;
		if (hnmfTest != NULL) delete hnmfTest;
		if (nmf != NULL) delete nmf;
	}

// Dialog Data
	enum { IDD = IDD_FACESNMF_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

private:	
	bool errorsLoadingImages;
	bool errorsLoadingImagesTest;

	void EnableIterationButtons();
	void LoadImages();
	void InvalidateImages();
	bool FolderHasImages(CString & folder, int & numberImages, int & imageWidth, int & imageHeight);
	void ErrorLoadingImage(char * error);
	bool LoadImage(CString & folder, int imageNumber, Matrix<cudafloat> & V);
	bool UpdateRank();

	void DrawPixel(FlickerFreeDC & dc, int x, int y, cudafloat colorValue, int zoom, bool inverted = false);

	int topMargin;
	int iteration;
	int iterationIncrease;

	void UpdateIteration(cudafloat quality = CUDA_VALUE(0.0), int totalIteractions = 0, int time = 0);
	void MoveWindow(CWnd & w, int top, int bottom);
	void Resize(int cx, int cy);
	void RescaleW();
	void CreateNMFobject(HostNMF * hnmf);
	void CreateNMFtrain();
	void CreateNMFtest();

	void Train(NMF * nmf, HostNMF * hnmf, cudafloat tol, bool updateW = true);
	void Train(HostNMF * hnmf, bool updateW = true);

	void CFacesNMFDlg::Save(Matrix<cudafloat> & m, const char * filename);

	CudaDevice cudaDevice;

	CComboBox comboMethod;
	CButton buttonRandomize;
	CStatic labelR;
	CEdit editRank;
	CEdit editTrainFolder;
	CEdit editTestFolder;	
	CEdit editTOL;
	CButton buttonIteractionTest;
	CButton buttonSave;
	CSliderCtrl sliderImage;
	CSliderCtrl sliderZoom;
	CSliderCtrl sliderIterations;
	CStatic labelIteration;
	CButton buttonIteration;
	CButton checkUseCUDA;

	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
	afx_msg void OnBnClickedOk();
	afx_msg void OnCbnSelchangeComboMethod();
	afx_msg void OnBnClickedButtonRnd();
	afx_msg void OnBnClickedButtonUpdateImages();
	afx_msg void OnBnClickedTrainTest();
	afx_msg void OnBnClickedButtonSave();
	
};