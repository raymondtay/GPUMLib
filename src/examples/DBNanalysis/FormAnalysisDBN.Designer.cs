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

namespace DBNanalysis {
	partial class FormAnalysisDBN {
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing) {
			if (disposing && (components != null)) {
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent() {
			this.folderBrowser = new System.Windows.Forms.FolderBrowserDialog();
			this.proprieties = new System.Windows.Forms.PropertyGrid();
			this.btLoadModel = new System.Windows.Forms.Button();
			this.splitContainer = new System.Windows.Forms.SplitContainer();
			this.imageScroll = new System.Windows.Forms.HScrollBar();
			this.btSaveOutputs = new System.Windows.Forms.Button();
			((System.ComponentModel.ISupportInitialize)(this.splitContainer)).BeginInit();
			this.splitContainer.Panel1.SuspendLayout();
			this.splitContainer.Panel2.SuspendLayout();
			this.splitContainer.SuspendLayout();
			this.SuspendLayout();
			// 
			// proprieties
			// 
			this.proprieties.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
			| System.Windows.Forms.AnchorStyles.Left) 
			| System.Windows.Forms.AnchorStyles.Right)));
			this.proprieties.Location = new System.Drawing.Point(0, 0);
			this.proprieties.Name = "proprieties";
			this.proprieties.Size = new System.Drawing.Size(267, 400);
			this.proprieties.TabIndex = 11;
			// 
			// btLoadModel
			// 
			this.btLoadModel.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
			this.btLoadModel.Location = new System.Drawing.Point(0, 406);
			this.btLoadModel.Name = "btLoadModel";
			this.btLoadModel.Size = new System.Drawing.Size(75, 23);
			this.btLoadModel.TabIndex = 12;
			this.btLoadModel.Text = "Load Model";
			this.btLoadModel.UseVisualStyleBackColor = true;
			this.btLoadModel.Click += new System.EventHandler(this.btLoadModel_Click);
			// 
			// splitContainer
			// 
			this.splitContainer.Dock = System.Windows.Forms.DockStyle.Fill;
			this.splitContainer.Location = new System.Drawing.Point(0, 0);
			this.splitContainer.Name = "splitContainer";
			// 
			// splitContainer.Panel1
			// 
			this.splitContainer.Panel1.Controls.Add(this.btSaveOutputs);
			this.splitContainer.Panel1.Controls.Add(this.btLoadModel);
			this.splitContainer.Panel1.Controls.Add(this.proprieties);
			// 
			// splitContainer.Panel2
			// 
			this.splitContainer.Panel2.Controls.Add(this.imageScroll);
			this.splitContainer.Panel2.Paint += new System.Windows.Forms.PaintEventHandler(this.DrawImages);
			this.splitContainer.Size = new System.Drawing.Size(696, 429);
			this.splitContainer.SplitterDistance = 275;
			this.splitContainer.TabIndex = 12;
			// 
			// imageScroll
			// 
			this.imageScroll.Dock = System.Windows.Forms.DockStyle.Bottom;
			this.imageScroll.LargeChange = 1;
			this.imageScroll.Location = new System.Drawing.Point(0, 412);
			this.imageScroll.Name = "imageScroll";
			this.imageScroll.Size = new System.Drawing.Size(417, 17);
			this.imageScroll.TabIndex = 1;
			this.imageScroll.ValueChanged += new System.EventHandler(this.imageScroll_ValueChanged);
			// 
			// btSaveOutputs
			// 
			this.btSaveOutputs.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
			this.btSaveOutputs.Location = new System.Drawing.Point(81, 406);
			this.btSaveOutputs.Name = "btSaveOutputs";
			this.btSaveOutputs.Size = new System.Drawing.Size(79, 23);
			this.btSaveOutputs.TabIndex = 13;
			this.btSaveOutputs.Text = "Save outputs";
			this.btSaveOutputs.UseVisualStyleBackColor = true;
			this.btSaveOutputs.Click += new System.EventHandler(this.btSaveOutputs_Click);
			// 
			// FormAnalysisDBN
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(696, 429);
			this.Controls.Add(this.splitContainer);
			this.Name = "FormAnalysisDBN";
			this.Text = "DBN Analysis";
			this.Load += new System.EventHandler(this.FormAnalysisDBN_Load);
			this.splitContainer.Panel1.ResumeLayout(false);
			this.splitContainer.Panel2.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.splitContainer)).EndInit();
			this.splitContainer.ResumeLayout(false);
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.FolderBrowserDialog folderBrowser;
		private System.Windows.Forms.PropertyGrid proprieties;
		private System.Windows.Forms.Button btLoadModel;
		private System.Windows.Forms.SplitContainer splitContainer;
		private System.Windows.Forms.HScrollBar imageScroll;
		private System.Windows.Forms.Button btSaveOutputs;
	}
}

