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

#include "stdafx.h"
#include "FacesNMF.h"
#include "FacesNMFDlg.h"
#include "definitions.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//#define INITIALIZE_RANDOM_GENERATOR srand(1246985095);
#define INITIALIZE_RANDOM_GENERATOR

#define SCREEN_UPDATE_INTERVAL (5000)


/**************
About dialog
**************/

class CAboutDlg : public CDialog {
public:
	CAboutDlg();

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()



/**************
Initialization
**************/

CFacesNMFDlg::CFacesNMFDlg(CWnd* pParent /*=NULL*/) : CDialog(CFacesNMFDlg::IDD, pParent) {
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	iteration = 0;
	iterationIncrease = 1;
	hnmf = NULL;
	hnmfTest = NULL;
	nmf = NULL;	
	rank = NMF_RANK;
	INITIALIZE_RANDOM_GENERATOR;
}

BOOL CFacesNMFDlg::OnInitDialog() {
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	editTrainFolder.SetWindowText("..\\images");

	editTOL.SetWindowText("0.01");

	ostringstream ss;
	ss << (rank);
	editRank.SetWindowText(ss.str().c_str());

	CRect r;
	GetClientRect(&r);
	Resize(r.right, r.bottom);

	sliderZoom.SetRange(1, NMF_MAX_IMAGE_ZOOM);
	sliderZoom.SetPos(NMF_INITIAL_IMAGE_ZOOM);

	sliderIterations.SetRangeMax(5);

	if (cudaDevice.SupportsCuda()) {
		checkUseCUDA.EnableWindow(TRUE);
		checkUseCUDA.SetCheck(BST_CHECKED);
	}

	nmfMethod = MULTIPLICATIVE_EUCLIDEAN;
	comboMethod.SetCurSel(nmfMethod);

	LoadImages();

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CFacesNMFDlg::DoDataExchange(CDataExchange* pDX) {
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER_IMAGE, sliderImage);
	DDX_Control(pDX, IDC_SLIDER_ZOOM, sliderZoom);
	DDX_Control(pDX, IDC_SLIDER_ITERATIONS, sliderIterations);
	DDX_Control(pDX, IDC_STATIC_ITERATION, labelIteration);
	DDX_Control(pDX, IDOK, buttonIteration);
	DDX_Control(pDX, IDC_CHECK_USE_CUDA, checkUseCUDA);
	DDX_Control(pDX, IDC_COMBO_METHOD, comboMethod);
	DDX_Control(pDX, IDC_BUTTON_RND, buttonRandomize);
	DDX_Control(pDX, IDC_STATIC_RANK, labelR);
	DDX_Control(pDX, IDC_EDIT_RANK, editRank);
	DDX_Control(pDX, IDC_EDIT_TRAIN_FOLDER, editTrainFolder);
	DDX_Control(pDX, IDC_EDIT_TEST_FOLDER, editTestFolder);	
	DDX_Control(pDX, IDC_EDIT_TOL, editTOL);
	DDX_Control(pDX, ID_TRAIN_TEST, buttonIteractionTest);
	DDX_Control(pDX, IDC_BUTTON_SAVE, buttonSave);
}

BEGIN_MESSAGE_MAP(CFacesNMFDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_WM_SIZE()
	ON_WM_HSCROLL()
	ON_BN_CLICKED(IDOK, &CFacesNMFDlg::OnBnClickedOk)
	ON_CBN_SELCHANGE(IDC_COMBO_METHOD, &CFacesNMFDlg::OnCbnSelchangeComboMethod)
	ON_BN_CLICKED(IDC_BUTTON_RND, &CFacesNMFDlg::OnBnClickedButtonRnd)
	ON_BN_CLICKED(IDC_BUTTON_UPDATE_IMAGES, &CFacesNMFDlg::OnBnClickedButtonUpdateImages)	
	ON_BN_CLICKED(ID_TRAIN_TEST, &CFacesNMFDlg::OnBnClickedTrainTest)
	ON_BN_CLICKED(IDC_BUTTON_SAVE, &CFacesNMFDlg::OnBnClickedButtonSave)
END_MESSAGE_MAP()

void CFacesNMFDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CFacesNMFDlg::OnQueryDragIcon() {
	return static_cast<HCURSOR>(m_hIcon);
}



/**************
Paint / Draw
**************/

void CFacesNMFDlg::DrawPixel(FlickerFreeDC & dc, int x, int y, cudafloat colorValue, int zoom, bool inverted) {
	colorValue *= CUDA_VALUE(255.0);
	if (colorValue > CUDA_VALUE(255.0)) colorValue = CUDA_VALUE(255.0);

	BYTE color = (BYTE) (colorValue);
	if (inverted) color = (BYTE) (255 - colorValue);
	COLORREF pixelColor = RGB(color, color, color);
	dc.FillSolidRect(x, y, zoom, zoom , pixelColor);
}

void CFacesNMFDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		FlickerFreeDC dc(this, RGB(240, 240, 240));

		CRect rect;
		GetClientRect(&rect);

		int zoom =  sliderZoom.GetPos();

		int imageWidthScreen = imageWidth * zoom + NMF_IMAGE_MARGIN;

		for (int i = 0; rect.right > i * imageWidthScreen; i++) {
			int pos = sliderImage.GetPos() + i;

			int image = (hnmf != NULL && pos < numberImages) ? pos : -1;
			int imageW = (hnmf != NULL && pos < rank) ? pos : -1;
			int imageT = (hnmfTest != NULL && pos < numberImagesTest) ?pos : -1;

			if (pos > image && pos > imageW && pos > imageT) break;

			SetBkMode(dc, TRANSPARENT);

			CString simg;
			simg.Format("%d", pos + 1);
			dc.TextOut((i * imageWidthScreen) + NMF_IMAGE_MARGIN, topMargin, simg);

			for(int r = 0; r < imageHeight; r++) {
				for(int c = 0; c < imageWidth; c++) {
					int b = r * imageHeight + c;
					
					int x = (i * imageWidthScreen) + c * zoom + NMF_IMAGE_MARGIN;
					int y = topMargin + r * zoom + 3 * NMF_IMAGE_MARGIN;

					if (image != -1) {
						// Train image (V)
						DrawPixel(dc, x, y, hnmf->V(b, image), zoom);

						// Aproximation to the train image (WH)
						y += imageHeight * zoom + NMF_IMAGE_MARGIN;
						DrawPixel(dc, x, y, hnmf->WH(b, image), zoom);
					}
					

					// parts (W)
					if  (imageW != -1) {
						y += imageHeight * zoom + NMF_IMAGE_MARGIN;
						DrawPixel(dc, x, y, Wrescaled(b, imageW), zoom);
						
						y += imageHeight * zoom + NMF_IMAGE_MARGIN;
						DrawPixel(dc, x, y, Wrescaled(b, imageW), zoom, true);
					}

					// Test images
					if (imageT != -1) {
						// Test image (Vtest)
						y += imageHeight * zoom + NMF_IMAGE_MARGIN;
						DrawPixel(dc, x, y, hnmfTest->V(b, imageT), zoom);

						// Aproximation to the test image (WHtest)
						y += imageHeight * zoom + NMF_IMAGE_MARGIN;
						DrawPixel(dc, x, y, hnmfTest->WH(b, imageT), zoom);
					}
				}
			}
		}

		CDialog::OnPaint();
	}
}



/**************
Load images
**************/

void CFacesNMFDlg::ErrorLoadingImage(char * error) {
	if (!errorsLoadingImages && !errorsLoadingImagesTest) AfxMessageBox(error);
}

bool CFacesNMFDlg::FolderHasImages(CString & folder, int & numberImages, int & imageWidth, int & imageHeight) {
	try {
		ostringstream ss;
		ss << folder << "\\" << "face00001.pgm";
		ifstream imageFile(ss.str().c_str(), ios::binary);

		string s;
		imageFile >> s;
		if (s!= "P5") {
			ErrorLoadingImage("Invalid image format");
			return false;
		}

		imageFile >> imageWidth >> imageHeight;
	} catch (...) {
		ErrorLoadingImage("Error loading images");
		return false;
	}

	numberImages = 0;
	int inc = 1024;	

	while(true) {
		CFileStatus fileStatus;

		ostringstream ss;
		ss << folder << "\\" << "face" << setfill('0') << setw(5) << (numberImages + inc) << ".pgm";

		if (!CFile::GetStatus(ss.str().c_str(), fileStatus)) {
			if (inc == 1) {
				break; 
			} else {
				inc >>= 1;
			}
		} else {
			numberImages += inc;
		}
	}

	return true;
}

void CFacesNMFDlg::LoadImages() {
	errorsLoadingImagesTest = errorsLoadingImages = false;

	UpdateRank();

	// train images

	numberImages = 0;

	CString folder;
	editTrainFolder.GetWindowText(folder);

	if (folder.IsEmpty() || !FolderHasImages(folder, numberImages, imageWidth, imageHeight)) {
		errorsLoadingImagesTest = errorsLoadingImages = true;
		return;
	}	

	int pixels = imageWidth * imageHeight;
	if (hnmf != NULL) delete hnmf;
	hnmf = new HostNMF(pixels, numberImages, rank);

	for (int i = 0; i < numberImages; i++) {
		if (!LoadImage(folder, i, hnmf->V)) errorsLoadingImages = true;
	}

	// test images
	numberImagesTest = 0;

	editTestFolder.GetWindowText(folder);

	int imageWidthTest, imageHeightTest;
	if (folder.IsEmpty() || !FolderHasImages(folder, numberImagesTest, imageWidthTest, imageHeightTest)) {
		errorsLoadingImagesTest = true;
	} else {		
		if (hnmfTest == NULL) delete hnmfTest;
		hnmfTest = new HostNMF(hnmf, numberImagesTest);

		for (int i = 0; i < numberImagesTest; i++) {
			if (!LoadImage(folder, i, hnmfTest->V)) errorsLoadingImagesTest = true;
		}
	}

	// update images slider and WH matrices

	sliderImage.SetRangeMax(((numberImages > numberImagesTest) ? numberImages : numberImagesTest) - 1);

	if (cudaDevice.SupportsCuda()) {
		CreateNMFtrain();
	} else {
		Matrix<cudafloat>::Multiply(hnmf->W, hnmf->H, hnmf->WH);
		RescaleW();
	}

	if (hnmfTest != NULL) Matrix<cudafloat>::Multiply(hnmfTest->W, hnmfTest->H, hnmfTest->WH);

	EnableIterationButtons();
}

bool CFacesNMFDlg::LoadImage(CString & folder, int imageNumber, Matrix<cudafloat> & V) {
	string s;
	int width;
	int height;
	int maxGrayValue;

	ostringstream ss;
	ss << folder << "\\" << "face" << setfill('0') << setw(5) << (imageNumber + 1) << ".pgm";
	try {
		ifstream imageFile(ss.str().c_str(), ios::binary);

		imageFile >> s;
		if (s!= "P5") {
			ErrorLoadingImage("Invalid image format");
			return false;
		}

		imageFile >> width >> height;
		if (width != imageWidth || height != imageHeight) {
			ErrorLoadingImage("Incorrect image size");
			return false;
		}

		imageFile >> maxGrayValue;
		if (maxGrayValue != 255) {
			ErrorLoadingImage("Check max gray value");
			return false;
		}

		int pixels = imageWidth * imageHeight;
		HostArray<char> buffer(pixels);
		imageFile.read(buffer.Pointer(), pixels);

		// histogram equalization
		float histogram[256];
		for(int i = 0; i < 256; i++) histogram[i] = 0;		
		for (int b = 0; b < pixels; b++) {
			int p = ((unsigned char) buffer[b]);
			histogram[p]++;
		}
		for(int i = 1; i < 256; i++) histogram[i] += histogram[i - 1];

		int i = 0;
		while (histogram[i] == 0) i++;
		float cdfmin = histogram[i];
		for(; i < 256; i++) {
			histogram[i] -= cdfmin;
			histogram[i] /= (pixels - cdfmin);
		}
		for (int b = 0; b < pixels; b++) V(b, imageNumber) = histogram[((unsigned char) buffer[b])];
	} catch (...) {
		ErrorLoadingImage("Error loading image");
		return false;
	}

	return true;
}



/**************
Resize
**************/

void CFacesNMFDlg::MoveWindow(CWnd & w, int top, int bottom) {
	CRect r;

	w.GetWindowRect(r);
	ScreenToClient(r);
	r.bottom = bottom;
	r.top = top;
	w.MoveWindow(r);
}

void CFacesNMFDlg::Resize(int cx, int cy) {
	CRect r;

	sliderImage.GetWindowRect(r);
	ScreenToClient(r);
	topMargin = r.bottom;
	r.right = cx - NMF_IMAGE_MARGIN;
	sliderImage.MoveWindow(r);
}

void CFacesNMFDlg::OnSize(UINT nType, int cx, int cy) {
	CRect r;

	CDialog::OnSize(nType, cx, cy);
	Resize(cx, cy);
}



/**************
Scroll bars
**************/

void CFacesNMFDlg::InvalidateImages() {
	CRect r;
	GetWindowRect(r);
	ScreenToClient(r);
	r.top = topMargin;

	InvalidateRect(r, FALSE);
}

void CFacesNMFDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	if (pScrollBar->GetSafeHwnd() == sliderIterations.GetSafeHwnd()) {
		iterationIncrease = 1;
		for (int i = 0; i < sliderIterations.GetPos(); i++) iterationIncrease *= 10;


		ostringstream ss;
		ss << "+" << (iterationIncrease) << " train";
		buttonIteration.SetWindowText(ss.str().c_str());

		ss.str("");
		ss << "+" << (iterationIncrease) << " test";
		buttonIteractionTest.SetWindowText(ss.str().c_str());
	} else {
		CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
		InvalidateImages();
	}
}



/**************
Iteration
**************/

void CFacesNMFDlg::EnableIterationButtons() {
	buttonIteration.EnableWindow(!errorsLoadingImages);
	buttonIteractionTest.EnableWindow(!errorsLoadingImagesTest);
}

void CFacesNMFDlg::UpdateIteration(cudafloat quality, int totalIteractions, int time) {
	ostringstream ss;
	ss << "Iteration: " << (iteration);

	if (iteration > 0 && quality >= CUDA_VALUE(0.0)) {
		ss << " | Quality improvement: " << quality;
	}

	if (totalIteractions > 0) {
		ss << " | " << totalIteractions << " iterations in " << time << " ms";
	}

	labelIteration.SetWindowText(ss.str().c_str());
	labelIteration.UpdateWindow();
}

void CFacesNMFDlg::Train(HostNMF * hnmf, bool updateW) {
	int initialTime = clock();
	for(int i=0; i < iterationIncrease; i++) {
		hnmf->DoIteration(nmfMethod, updateW);
		iteration++;
	}

	int time = (clock() - initialTime);

	Matrix<cudafloat>::Multiply(hnmf->W, hnmf->H, hnmf->WH);

	UpdateIteration(CUDA_VALUE(-1.0), iterationIncrease, time);
}

void CFacesNMFDlg::Train(NMF * nmf, HostNMF * hnmf, cudafloat tol, bool updateW) {
	cudafloat previousQuality = nmf->QualityImprovement();

	int initialTime = clock();
	int initialIterations = iteration;
	int lastScreenUpdate = initialTime;

	for(int i=0; i < iterationIncrease; i++) {
		nmf->DoIteration(updateW);
		iteration++;
			
		cudafloat quality = nmf->QualityImprovement();
		if (quality != previousQuality) {
			if (quality < tol) break;

			previousQuality = quality;

			int time = clock();
			if (time - lastScreenUpdate >= SCREEN_UPDATE_INTERVAL) {
				lastScreenUpdate = time;
				UpdateIteration(quality);
			}
		}			
	}

	cudaThreadSynchronize();
	UpdateIteration(nmf->QualityImprovement(), iteration - initialIterations, clock() - initialTime);

	hnmf->W = nmf->GetW();
	hnmf->H = nmf->GetH();
	hnmf->WH = nmf->GetWH();
}


void CFacesNMFDlg::OnBnClickedOk() {
	CCmdTarget::BeginWaitCursor();

	buttonIteration.EnableWindow(FALSE);
	buttonIteractionTest.EnableWindow(FALSE);

	UpdateRank();

	CString stol;
	editTOL.GetWindowText(stol);
	cudafloat tol = (cudafloat) atof(stol);

	if (checkUseCUDA.GetCheck() == BST_CHECKED) {
		if (nmf == NULL || !nmfContainsTrainData) CreateNMFtrain();
		Train(nmf, hnmf, tol, true);
	} else {
		Train(hnmf);
	}

	if (hnmfTest != NULL) {
		hnmfTest->W = hnmf->W;
		Matrix<cudafloat>::Multiply(hnmfTest->W, hnmfTest->H, hnmfTest->WH);
	}

	RescaleW();
	InvalidateImages();

	CCmdTarget::EndWaitCursor();
	EnableIterationButtons();
}

void CFacesNMFDlg::OnBnClickedTrainTest() {
	CCmdTarget::BeginWaitCursor();

	buttonIteration.EnableWindow(FALSE);
	buttonIteractionTest.EnableWindow(FALSE);

	UpdateRank();

	CString stol;
	editTOL.GetWindowText(stol);
	cudafloat tol = (cudafloat) atof(stol);

	if (checkUseCUDA.GetCheck() == BST_CHECKED) {
		if (nmf == NULL || nmfContainsTrainData) CreateNMFtest();
		Train(nmf, hnmfTest, tol, false);
	} else {
		Train(hnmfTest, false);
	}

	Matrix<cudafloat>::Multiply(hnmfTest->W, hnmfTest->H, hnmfTest->WH);

	InvalidateImages();

	CCmdTarget::EndWaitCursor();
	EnableIterationButtons();
}



/**************
NMF
**************/


void CFacesNMFDlg::RescaleW() {
	cudafloat minimum;
	cudafloat maximum;

	Wrescaled = hnmf->W;

	for(int r = 0; r < Wrescaled.Rows(); r++) {
		minimum = maximum = Wrescaled(r, 0);

		for(int c = 1; c < Wrescaled.Columns(); c++) {
			cudafloat value = Wrescaled(r, c);

			if (value < minimum) {
				minimum = value;
			} else if (value > maximum) {
				maximum = value;
			}
		}

		for(int c = 0; c < Wrescaled.Columns(); c++) {
			Wrescaled(r, c) = (minimum == maximum) ? CUDA_VALUE(0.0) : (Wrescaled(r, c) - minimum) / (maximum - minimum);
		}
	}
}

void CFacesNMFDlg::CreateNMFtrain() {
	nmfContainsTrainData = true;
	CreateNMFobject(hnmf);
	hnmf->WH = nmf->GetWH();
	RescaleW();
}

void CFacesNMFDlg::CreateNMFtest() {
	nmfContainsTrainData = false;
	CreateNMFobject(hnmfTest);
}

void CFacesNMFDlg::CreateNMFobject(HostNMF * hnmf) {
	if (nmf != NULL) delete nmf;

	switch (nmfMethod) {
		case MULTIPLICATIVE_EUCLIDEAN:
			nmf = new NMF_MultiplicativeEuclidianDistance(hnmf->V, hnmf->W, hnmf->H);
			break;

		case MULTIPLICATIVE_DIVERGENCE:
			nmf = new NMF_MultiplicativeDivergence(hnmf->V, hnmf->W, hnmf->H);
			break;

		case ADDITIVE_EUCLIDEAN:
			nmf = new NMF_AdditiveEuclidian(hnmf->V, hnmf->W, hnmf->H);
			break;

		case ADDITIVE_DIVERGENCE:
			nmf = new NMF_AdditiveDivergence(hnmf->V, hnmf->W, hnmf->H);
			break;
	}
}



/**************
Other Buttons and control events
**************/

bool CFacesNMFDlg::UpdateRank() {
	CString s;
	editRank.GetWindowText(s);
	int r = atoi(s);

	if (r == rank) return false;

	rank = r;

	if (hnmf != NULL) {
		hnmf->SetR(r);
		if (hnmfTest != NULL) hnmfTest->SetR(r);

		if (cudaDevice.SupportsCuda()) {
			CreateNMFtrain();
		} else {
			Matrix<cudafloat>::Multiply(hnmf->W, hnmf->H, hnmf->WH);
			RescaleW();
		}

		if (hnmfTest != NULL) {
			hnmfTest->W = hnmf->W;
			Matrix<cudafloat>::Multiply(hnmfTest->W, hnmfTest->H, hnmfTest->WH);
		}
	}

	iteration = 0;
	UpdateIteration();
	InvalidateImages();
	
	return true;
}

void CFacesNMFDlg::OnCbnSelchangeComboMethod() {
	NMF_METHOD m = (NMF_METHOD) comboMethod.GetCurSel();

	if (m == nmfMethod) {
		UpdateRank();
	} else {
		nmfMethod = m;
		if (!UpdateRank() && cudaDevice.SupportsCuda() && hnmf != NULL) {
			if (nmfContainsTrainData) {
				CreateNMFtrain();
			} else {
				CreateNMFtest();
			}
		}
	}
}

void CFacesNMFDlg::OnBnClickedButtonRnd() {
	if (UpdateRank()) return;

	if (hnmf != NULL) {
		hnmf->Randomize();
		if (hnmfTest != NULL) hnmfTest->Randomize();

		if (cudaDevice.SupportsCuda()) {
			CreateNMFtrain();
		} else {
			Matrix<cudafloat>::Multiply(hnmf->W, hnmf->H, hnmf->WH);
			RescaleW();
		}

		if (hnmfTest != NULL) {
			hnmfTest->W = hnmf->W;
			Matrix<cudafloat>::Multiply(hnmfTest->W, hnmfTest->H, hnmfTest->WH);
		}
	}

	iteration = 0;
	UpdateIteration();
	InvalidateImages();
}

void CFacesNMFDlg::OnBnClickedButtonUpdateImages() {
	iteration = 0;
	UpdateIteration();

	LoadImages();
	InvalidateImages();
}

void CFacesNMFDlg::Save(Matrix<cudafloat> & m, const char * filename) {
	ofstream f(filename, ios::out);

	f << fixed << setprecision (PRECISION);

	int columns = m.Columns();
	int rows = m.Rows();
	for(int i = 0; i < columns; i++) {
		for(int r = 0; r < rows; ) {
			f << m(r, i);
			if (++r < rows) f << ",";
		}
		f << endl;
	}
}

void CFacesNMFDlg::OnBnClickedButtonSave() {
	if (hnmf == NULL) return;

	Save(hnmf->H, "Htrain.csv");
	Save(hnmf->W, "W.csv");
	
	if (hnmfTest != NULL) Save(hnmfTest->H, "Htest.csv");
}
