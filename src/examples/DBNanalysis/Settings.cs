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

using System.ComponentModel;
using System.Drawing.Design;

namespace DBNanalysis {
	[DefaultPropertyAttribute("ModelFilename")]
	class Settings {
		private int zoom = 1;

		[CategoryAttribute("Model"), DescriptionAttribute("File name containing the DBN model"), DisplayName("File name"), Editor(typeof(ModelFilenameEditor), typeof(UITypeEditor))]
		public string ModelFilename { get; set; }

		[CategoryAttribute("Images"), DescriptionAttribute("Images width"), DefaultValue(0)]
		public int Width { get; set; }

		[CategoryAttribute("Images"), DescriptionAttribute("Images height"), DefaultValue(0)]
		public int Height { get; set; }

		[CategoryAttribute("Images"), DescriptionAttribute("Zoom"), DefaultValue(1)]
		public int Zoom {
			get { return zoom; }
			set { if (value >= 1) zoom = value; }
		}

		[CategoryAttribute("Reconstruction"), DescriptionAttribute("K steps"), DefaultValue(1)]
		public int K { get; set; }

		[CategoryAttribute("Reconstruction"), DescriptionAttribute("Layers to process"), DisplayName("Layers to process"), DefaultValue(int.MaxValue)]
		public int LayersToProcess { get; set; }

		public Settings() {
			LayersToProcess = int.MaxValue;
			Width = Height = 0;
			K = 1;
		}
	}
}
