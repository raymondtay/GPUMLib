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

using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using System.Xml;
using System.Xml.Serialization;

namespace DBNanalysis {
	public partial class FormAnalysisDBN : Form {
		private const int SPACE_BETWEEN_IMAGES = 4;
		private const int RECEPTIVE_FIELDS_ROWS = 10;

		private static Color BACKGROUNG_COLOR = Color.White;

		private static Random random = new Random();

		private float minWeight;
		private float maxWeight;

		private float maxWeightBias;

		private DBNmodel model = null;
		private DataSet dstrain = null;
		private Settings settings;

		public FormAnalysisDBN() {
			InitializeComponent();
		}

		private void FormAnalysisDBN_Load(object sender, EventArgs e) {
			settings = new Settings();
			proprieties.SelectedObject = settings;
		}

		private void btLoadModel_Click(object sender, EventArgs e) {
			try {
				Cursor.Current = Cursors.WaitCursor;
				XmlSerializer xmlSerializer = new XmlSerializer(typeof(DBNmodel));
				XmlReader xmlReader = XmlReader.Create(settings.ModelFilename);

				model = (DBNmodel)xmlSerializer.Deserialize(xmlReader);
			} catch {
				model = null;
			} finally {
				Cursor.Current = Cursors.Arrow;
			}

			if (model == null || model.layers.Length == 0) {
				MessageBox.Show("Invalid model.");
				return;
			}

			RBMlayer bottomLayer = model.layers[0];

			int visibleUnits = bottomLayer.I;

			minWeight = maxWeight = bottomLayer.weights[0];
			maxWeightBias = 0.0f;

			for (int j = 0; j < bottomLayer.J; j++) {
				float bias = bottomLayer.biasHiddenLayer[j];

				for (int i = 0; i < visibleUnits; i++) {
					float w = bottomLayer.Weight(j, i);

					if (minWeight > w) {
						minWeight = w;
					} else if (maxWeight < w) {
						maxWeight = w;
					}

					if (bias + w > maxWeightBias) maxWeightBias = bias + w;
				}
			}

			if (settings.Width == 0 && settings.Height == 0) {
				int pixels = (int)Math.Sqrt(visibleUnits);
				if (visibleUnits == pixels * pixels) {
					settings.Width = settings.Height = pixels;
					proprieties.SelectedObject = settings;
				}
			}

			dstrain = new DataSet(model.TrainFilename, Path.GetDirectoryName(settings.ModelFilename));
			imageScroll.Maximum = dstrain.Samples - 1;

			splitContainer.Panel2.Invalidate();
		}

		private void DrawImages(object sender, PaintEventArgs e) {
			if (model == null || settings.Width == 0 || settings.Height == 0) return;

			SplitterPanel panelImages = (SplitterPanel)sender;

			Bitmap img = new Bitmap(panelImages.Width, panelImages.Height);
			Graphics g = Graphics.FromImage(img);

			g.Clear(BACKGROUNG_COLOR);

			int widthFactor = settings.Zoom * settings.Width + SPACE_BETWEEN_IMAGES;

			for (int i = imageScroll.Value; i < dstrain.Samples; i++) {
				int x = (i - imageScroll.Value) * widthFactor;
				if (x >= panelImages.Width) break;

				DrawSample(g, i, x);
			}

			e.Graphics.DrawImage(img, 0, 0);
		}

		private void DrawSample(Graphics graphics, int i, int xOrigin) {
			int yOrigin = SPACE_BETWEEN_IMAGES;

			float[] imgBytes = dstrain[i];
			DrawSampleImage(graphics, xOrigin, yOrigin, imgBytes);

			for (int k = 0; k < settings.K; k++) {
				if (k > 0) for (int b = 0; b < imgBytes.Length; b++) imgBytes[b] = FormAnalysisDBN.Binarize(imgBytes[b]);

				imgBytes = Reconstruction(imgBytes);
				yOrigin += settings.Height * settings.Zoom + SPACE_BETWEEN_IMAGES;
				DrawSampleImage(graphics, xOrigin, yOrigin, imgBytes);
			}
		}

		private void DrawSampleImage(Graphics graphics, int xOrigin, int yOrigin, float[] imgBytes) {
			for (int h = 0; h < settings.Height; h++) {
				for (int w = 0; w < settings.Width; w++) {
					int color = (int)(255.0f * imgBytes[h * settings.Width + w]);
					int y = yOrigin + h * settings.Zoom;
					int x = xOrigin + w * settings.Zoom;

					graphics.FillRectangle(new SolidBrush(Color.FromArgb(color, color, color)), x, y, settings.Zoom, settings.Zoom);
				}
			}
		}

		private void imageScroll_ValueChanged(object sender, EventArgs e) {
			splitContainer.Panel2.Invalidate();
		}

		private static double Sigmoid(float x) {
			return 1.0 / (1.0 + Math.Exp(-x));
		}

		private static float Binarize(double probability) {
			return (probability > random.NextDouble()) ? 1.0f : 0.0f;
		}

		float[] Reconstruction(float[] originalValues) {
			float[] prevLayerValues = originalValues;

			int currentLayer = 0;
			foreach (RBMlayer layer in model.layers) {
				float[] layerValues = new float[layer.J];

				for (int j = 0; j < layer.J; j++) {
					float sum = layer.biasHiddenLayer[j];
					for (int i = 0; i < layer.I; i++) sum += prevLayerValues[i] * layer.Weight(j, i);
					layerValues[j] = Binarize(Sigmoid(sum));
				}

				prevLayerValues = layerValues;

				if (++currentLayer >= settings.LayersToProcess) break;
			}

			for (int l = currentLayer - 1; l >= 0; l--) {
				RBMlayer layer = model.layers[l];
				float[] layerValues = new float[layer.I];

				for (int i = 0; i < layer.I; i++) {
					float sum = layer.biasVisibleLayer[i];
					for (int j = 0; j < layer.J; j++) sum += prevLayerValues[j] * layer.Weight(j, i);
					layerValues[i] = (float)Sigmoid(sum);
				}

				prevLayerValues = layerValues;
			}

			return prevLayerValues;
		}

		private void btSaveOutputs_Click(object sender, EventArgs e) {
			StreamWriter writer = null;

			int layersToProcess = model.layers.Length;
			if (settings.LayersToProcess < layersToProcess) layersToProcess = settings.LayersToProcess;

			try {
				writer = new StreamWriter(string.Format("layer{0}-output.txt", layersToProcess - 1));

				for (int s = 0; s < dstrain.Samples; s++) {
					float[] prevLayerValues = dstrain[s];

					for (int l = 0; l < layersToProcess; l++) {
						RBMlayer layer = model.layers[l];

						float[] layerValues = new float[layer.J];

						for (int j = 0; j < layer.J; j++) {
							float sum = layer.biasHiddenLayer[j];
							for (int i = 0; i < layer.I; i++) sum += prevLayerValues[i] * layer.Weight(j, i);
							layerValues[j] = (Sigmoid(sum) >= 0.5) ? 1.0f : 0.0f;
						}

						prevLayerValues = layerValues;
					}

					foreach (float v in prevLayerValues) writer.Write("{0} ", v);
					writer.WriteLine();
				}
			} catch (System.Exception exception) {
				throw exception;
			} finally {
				if (writer != null) writer.Close();
			}
		}
	}
}

