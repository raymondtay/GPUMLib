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
using System.Drawing.Design;
using System.Windows.Forms;

namespace DBNanalysis {
	class ModelFilenameEditor : UITypeEditor {
		public override UITypeEditorEditStyle GetEditStyle(System.ComponentModel.ITypeDescriptorContext context) {
			return UITypeEditorEditStyle.Modal;
		}

		public override object EditValue(System.ComponentModel.ITypeDescriptorContext context, IServiceProvider provider, object value) {
			OpenFileDialog fileDialog = new OpenFileDialog();

			fileDialog.Filter = "DBN model files|*.dbn|All files|*.*";
			if (value != null) fileDialog.FileName = value.ToString();

			if (fileDialog.ShowDialog() == DialogResult.OK) return fileDialog.FileName;

			return base.EditValue(context, provider, value);
		}
	}
}
