import sys
import os
import glob
import torch
import torchvision
import numpy as np
import pydicom
import torchxrayvision as xrv
import torch.nn.functional as F
import pandas as pd

from torchcam.methods import SmoothGradCAMpp
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QFileDialog, QTextEdit, QMessageBox, QComboBox, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QScrollArea, QWidget


class XrayApp(QWidget):
    def __init__(self):
        super().__init__()
        self.folder_path = ""
        self.model = None
        self.selected_weight = None
        self.selected_pathology = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Chest X-ray Classification")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        self.folder_btn = QPushButton("Select Folder (DICOMs)", self)
        self.folder_btn.clicked.connect(self.load_folder)
        layout.addWidget(self.folder_btn)

        # Dropdown for weights
        self.weight_label = QLabel("Select pretrained weights:")
        layout.addWidget(self.weight_label)

        self.weight_dropdown = QComboBox(self)
        self.weight_dropdown.addItems([
            "",  # empty option at top
            "densenet121-res224-chex",
            "densenet121-res224-all",
            "densenet121-res224-rsna",
            "densenet121-res224-nih",
            "densenet121-res224-pc",
            "densenet121-res224-mimic_nb",
            "densenet121-res224-mimic_ch"
        ])
        self.weight_dropdown.currentTextChanged.connect(self.set_weight)
        layout.addWidget(self.weight_dropdown)

        # Dropdown for pathologies (populated after model loads)
        self.pathology_label = QLabel("Select pathology to check:")
        layout.addWidget(self.pathology_label)

        self.pathology_dropdown = QComboBox(self)
        layout.addWidget(self.pathology_dropdown)

        self.run_btn = QPushButton("Run Classification", self)
        self.run_btn.clicked.connect(self.run_classification)
        layout.addWidget(self.run_btn)

        self.reset_btn = QPushButton("Reset", self)
        self.reset_btn.clicked.connect(self.reset_app)
        layout.addWidget(self.reset_btn)        

        # Scrollable image viewer
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        layout.addWidget(self.scroll_area, stretch=2)        

        # Progress bar
        self.progress = QProgressBar(self)
        layout.addWidget(self.progress)

        # Text output
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.setLayout(layout)

    def load_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.folder_path:
            # reset outputs
            self.output_text.clear()
            self.progress.setValue(0)

            # clear scroll area images
            while self.scroll_layout.count():
                child = self.scroll_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            self.output_text.append(f"Selected folder: {self.folder_path}")

    def set_weight(self, weight_name):
        if weight_name.strip() == "":
            return  # ignore empty selection
        self.selected_weight = weight_name
        self.output_text.append(f"Selected weights: {self.selected_weight}")

        # Load model when weight is selected
        self.model = xrv.models.DenseNet(weights=self.selected_weight)
        self.output_text.append(f"Model loaded with {self.selected_weight}")

        # Populate pathologies dropdown (remove empty entries)
        clean_pathologies = [p for p in self.model.pathologies if p.strip() != ""]
        self.pathology_dropdown.clear()
        self.pathology_dropdown.addItems(clean_pathologies)

        # Reset selected pathology
        self.selected_pathology = None
        self.pathology_dropdown.currentTextChanged.connect(self.set_pathology)

    def set_pathology(self, pathology_name):
        if pathology_name.strip():
            self.selected_pathology = pathology_name
            self.output_text.append(f"Selected pathology: {self.selected_pathology}")

    def reset_app(self):
        # Clear text output
        self.output_text.clear()

        # Reset progress
        self.progress.setValue(0)

        # Clear images from scroll area
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Reset dropdown selections
        self.weight_dropdown.setCurrentIndex(0)
        self.pathology_dropdown.clear()

        # Reset variables
        self.folder_path = ""
        self.model = None
        self.selected_weight = None
        self.selected_pathology = None

        QMessageBox.information(self, "Reset", "Application has been reset.")


    def run_classification(self):
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "Please select a folder first!")
            return
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please select model weights first!")
            return
        if not self.selected_pathology:
            QMessageBox.warning(self, "Error", "Please select a pathology!")
            return

        pathology_idx = self.model.pathologies.index(self.selected_pathology)
        dcm_files = glob.glob(os.path.join(self.folder_path, "**", "*.dcm"), recursive=True)

        if not dcm_files:
            QMessageBox.warning(self, "Error", "No DICOM files found in folder!")
            return

        self.progress.setMaximum(len(dcm_files))
        self.progress.setValue(0)

        total, correct, incorrect = 0, 0, 0
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

        results = []
        save_dir = os.path.join(self.folder_path, "results")
        os.makedirs(save_dir, exist_ok=True)        
        for idx, dcm_file in enumerate(dcm_files):
            try:
                img_ds = pydicom.dcmread(dcm_file)
                img = img_ds.pixel_array.astype(np.float32)
                if img_ds.PhotometricInterpretation == "MONOCHROME1":
                    img = np.max(img) - img

                img_norm = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = img_norm.astype(np.uint8)
                img = xrv.datasets.normalize(img, 255)
                if img.ndim == 3:
                    img = img.mean(axis=2)
                img = img[None, ...]
                img = transform(img)
                img = torch.from_numpy(img)

                # CAM extractor
                cam_extractor = SmoothGradCAMpp(self.model, target_layer="features.norm5")

                scores = self.model(img[None, ...])
                prob = scores[0][pathology_idx].item()
                flag = prob > 0.5

                if flag:
                    correct += 1
                else:
                    incorrect += 1
                total += 1

                top3_idx = torch.argsort(scores.squeeze(), descending=True)[:3].tolist()
                top3_classes = [(self.model.pathologies[i], scores.squeeze()[i].item()) for i in top3_idx]
                # record structured result
                results.append({
                    "File": os.path.basename(dcm_file),
                    "Classified": "Yes" if flag else "No",
                    "Associated_Disease_1": f"{top3_classes[0][0]} ({top3_classes[0][1]:.4f})",
                    "Associated_Disease_2": f"{top3_classes[1][0]} ({top3_classes[1][1]:.4f})",
                    "Associated_Disease_3": f"{top3_classes[2][0]} ({top3_classes[2][1]:.4f})",
                })

                # log to text widget (structured)
                self.output_text.append(
                    f"File: {os.path.basename(dcm_file)}\n"
                    f"Classified as {self.selected_pathology}: {'Yes' if flag else 'No'}\n\nAssociated other diseases\n"
                    f"  1) {top3_classes[0][0]} ({top3_classes[0][1]:.4f})\n"
                    f"  2) {top3_classes[1][0]} ({top3_classes[1][1]:.4f})\n"
                    f"  3) {top3_classes[2][0]} ({top3_classes[2][1]:.4f})\n"
                    "-------------------------------------------------------------\n"
                )

                # CAM overlay
                activation_map = cam_extractor(class_idx=pathology_idx, scores=scores)
                cam_resized = F.interpolate(
                    torch.cat(activation_map).unsqueeze(0),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                ).squeeze()


                # --- Visualization update ---
                fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
                ax.imshow(img[None,...].squeeze().cpu(), cmap="gray")
                ax.imshow(cam_resized, cmap="jet", alpha=0.5)
                ax.axis("off")
                fig.canvas.draw()

                # Convert matplotlib figure to QPixmap
                w, h = fig.canvas.get_width_height()
                # buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
                # qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888)
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                qimg = QImage(buf.data, w, h, 4 * w, QImage.Format_RGBA8888)                
                pixmap = QPixmap.fromImage(qimg)

                container = QWidget()
                vbox = QVBoxLayout(container)
                # Title label (filename + flag + top3)
                title_label = QLabel(
                    f"{os.path.basename(dcm_file)} → {flag} | "
                    + ", ".join([f"{cls} ({score:.2f})" for cls, score in top3_classes])
                )
                title_label.setAlignment(Qt.AlignCenter)

                # Image label
                img_label = QLabel(self)
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setPixmap(pixmap.scaled(
                    500, 500,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

                # Add to container
                vbox.addWidget(title_label)
                vbox.addWidget(img_label)

                # Append to scroll area
                self.scroll_layout.addWidget(container)

                plt.close(fig)     

                # Update progress bar
                self.progress.setValue(idx + 1)
                QApplication.processEvents()

            except Exception as e:
                self.output_text.append(f"Error {dcm_file}: {e}")
                continue


        # save CSV
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(save_dir, str(self.selected_pathology)+"_Classification_results.csv")
            df.to_csv(csv_path, index=False)
            self.output_text.append(f"Results saved to {csv_path}")
        if total > 0:
            self.output_text.append(f"Accuracy: {correct/total:.2f} ({correct}/{total})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XrayApp()
    window.show()
    sys.exit(app.exec_())
