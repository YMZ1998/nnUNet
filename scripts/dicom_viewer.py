"""
DICOM 浏览器 - 支持 DICOM 系列和 RTSTRUCT ROI 显示
功能：
- DICOM 系列读取与排序（按 ImagePositionPatient）
- 窗宽/窗位调节
- 缩放、平移、切片浏览
- RTSTRUCT ROI 显示（3D 轮廓投影到切片）
- 支持 16-bit 值域
- 导出二值 mask
"""

import os
import sys
import traceback
from typing import Dict, List, Tuple, Optional

import SimpleITK as sitk
import numpy as np
import pydicom
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QWheelEvent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QLabel, QPushButton, QSlider, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QCheckBox, QColorDialog,
    QDoubleSpinBox, QSplitter, QListWidgetItem, QProgressDialog,
    QComboBox
)
from pydicom.dataset import Dataset

try:
    from skimage.draw import polygon

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("警告: scikit-image 未安装，导出 mask 功能将受限")


class DicomImageView(QLabel):
    """自定义 DICOM 图像显示控件，支持缩放、平移"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 800)  # 增加最小尺寸
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: black; }")

        # 显示参数
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.last_mouse_pos = None
        self.contour_line_width = 1.0  # ROI 轮廓线宽度

        # 图像数据
        self.current_image = None  # numpy array (原始大小)
        self.current_qimage = None
        self.rois_to_draw = []  # List of (contours, color, filled)
        self.original_size = None  # 原始图像尺寸
        self.use_smooth_transform = True  # 使用平滑变换

    def setImage(self, image_array: np.ndarray, rois: list = None):
        """设置要显示的图像（8-bit numpy array）"""
        if image_array is None:
            self.current_image = None
            self.current_qimage = None
            self.original_size = None
            self.clear()
            return

        self.current_image = image_array
        self.rois_to_draw = rois if rois is not None else []
        self.original_size = image_array.shape

        # 转换为 QImage（保持原始分辨率）
        height, width = image_array.shape
        bytes_per_line = width

        # 确保数据连续性
        image_array_copy = np.ascontiguousarray(image_array)

        self.current_qimage = QImage(
            image_array_copy.data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )

        self.updateDisplay()

    def updateDisplay(self):
        """更新显示 - 高分辨率渲染"""
        if self.current_qimage is None:
            return

        # 获取控件大小和原始图像大小
        widget_size = self.size()
        image_width = self.current_qimage.width()
        image_height = self.current_qimage.height()

        # 计算适应控件的缩放比例（保持原始分辨率）
        scale_to_fit = min(
            widget_size.width() / image_width,
            widget_size.height() / image_height
        ) * 0.95  # 留一点边距

        # 总缩放比例 = 适应缩放 × 用户缩放
        total_scale = scale_to_fit * self.zoom_factor

        # 计算目标尺寸（高分辨率）
        target_width = int(image_width * total_scale)
        target_height = int(image_height * total_scale)

        # 创建高分辨率 pixmap
        pixmap = QPixmap.fromImage(self.current_qimage)

        # 先缩放到目标尺寸（高质量）
        transform_mode = Qt.SmoothTransformation if self.use_smooth_transform else Qt.FastTransformation

        scaled_pixmap = pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            transform_mode  # 根据设置选择变换模式
        )

        # 在缩放后的 pixmap 上绘制 ROI
        if self.rois_to_draw:
            painter = QPainter(scaled_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.setRenderHint(QPainter.HighQualityAntialiasing)

            # ROI 坐标也需要缩放
            roi_scale = total_scale

            for roi_data in self.rois_to_draw:
                contours = roi_data['contours']
                color = roi_data['color']
                filled = roi_data['filled']
                opacity = roi_data['opacity']
                line_width = roi_data.get('line_width', 1.0)

                # 设置颜色和透明度
                qcolor = QColor(color[0], color[1], color[2], int(opacity * 255))

                if filled:
                    painter.setBrush(QBrush(qcolor))
                    painter.setPen(Qt.NoPen)
                else:
                    painter.setBrush(Qt.NoBrush)
                    pen = QPen(qcolor)
                    pen.setWidthF(max(0.5, line_width * roi_scale))  # 使用自定义线宽，线宽随缩放调整
                    pen.setCosmetic(False)  # 线宽随变换缩放
                    painter.setPen(pen)

                # 绘制每个轮廓（缩放坐标）
                for contour in contours:
                    if len(contour) < 2:
                        continue
                    from PyQt5.QtGui import QPolygonF
                    # 将轮廓坐标缩放到显示尺寸
                    polygon = QPolygonF([
                        QPointF(float(p[0]) * roi_scale, float(p[1]) * roi_scale)
                        for p in contour
                    ])
                    painter.drawPolygon(polygon)

            painter.end()

        self.setPixmap(scaled_pixmap)

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1

        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        self.updateDisplay()

    def mousePressEvent(self, event):
        """鼠标按下"""
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and
                                                 event.modifiers() & Qt.ControlModifier):
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """鼠标移动 - 平移"""
        if self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += QPointF(delta.x(), delta.y())
            self.last_mouse_pos = event.pos()
            # 平移功能暂时通过改变 pixmap 位置实现（简化版）

    def mouseReleaseEvent(self, event):
        """鼠标释放"""
        self.last_mouse_pos = None

    def resetView(self):
        """重置视图"""
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.updateDisplay()


class DicomSeriesLoader:
    """DICOM 系列加载器"""

    @staticmethod
    def load_dicom_series(folder_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict], List[Dataset]]:
        """
        加载 DICOM 系列
        返回: (volume_array, metadata, dicom_slices)
        """
        try:
            # 读取文件夹中所有 DICOM 文件
            dicom_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        ds = pydicom.dcmread(filepath, force=True)
                        if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'PixelData'):
                            dicom_files.append((filepath, ds))
                    except:
                        continue

            if not dicom_files:
                return None, None, []

            # 按 ImagePositionPatient 的 Z 坐标排序
            dicom_files.sort(key=lambda x: float(x[1].ImagePositionPatient[2]))

            slices = [ds for _, ds in dicom_files]

            # 提取元数据
            first_slice = slices[0]
            metadata = {
                'spacing': DicomSeriesLoader._get_spacing(slices),
                'origin': np.array([float(x) for x in first_slice.ImagePositionPatient]),
                'direction': DicomSeriesLoader._get_direction(first_slice),
                'rescale_slope': float(getattr(first_slice, 'RescaleSlope', 1.0)),
                'rescale_intercept': float(getattr(first_slice, 'RescaleIntercept', 0.0)),
                'num_slices': len(slices),
                'rows': int(first_slice.Rows),
                'cols': int(first_slice.Columns),
            }

            # 构建 3D 体积
            volume = DicomSeriesLoader._build_volume(slices, metadata)

            return volume, metadata, slices

        except Exception as e:
            print(f"加载 DICOM 系列失败: {e}")
            traceback.print_exc()
            return None, None, []

    @staticmethod
    def _get_spacing(slices: List[Dataset]) -> np.ndarray:
        """获取体素间距"""
        pixel_spacing = slices[0].PixelSpacing

        if len(slices) > 1:
            # 计算切片间距
            pos1 = np.array([float(x) for x in slices[0].ImagePositionPatient])
            pos2 = np.array([float(x) for x in slices[1].ImagePositionPatient])
            slice_spacing = np.linalg.norm(pos2 - pos1)
        else:
            slice_spacing = float(getattr(slices[0], 'SliceThickness', 1.0))

        return np.array([slice_spacing, float(pixel_spacing[0]), float(pixel_spacing[1])])

    @staticmethod
    def _get_direction(ds: Dataset) -> np.ndarray:
        """获取方向矩阵"""
        if hasattr(ds, 'ImageOrientationPatient'):
            iop = [float(x) for x in ds.ImageOrientationPatient]
            row_cosine = np.array(iop[:3])
            col_cosine = np.array(iop[3:6])
            slice_cosine = np.cross(row_cosine, col_cosine)

            direction = np.eye(3)
            direction[:, 0] = row_cosine
            direction[:, 1] = col_cosine
            direction[:, 2] = slice_cosine
            return direction
        else:
            return np.eye(3)

    @staticmethod
    def _build_volume(slices: List[Dataset], metadata: dict) -> np.ndarray:
        """构建 3D 体积数组"""
        num_slices = len(slices)
        rows = metadata['rows']
        cols = metadata['cols']

        # 获取原始数据类型
        first_array = slices[0].pixel_array

        # 创建体积数组
        volume = np.zeros((num_slices, rows, cols), dtype=np.float32)

        for i, ds in enumerate(slices):
            pixel_array = ds.pixel_array.astype(np.float32)

            # 应用 RescaleSlope 和 RescaleIntercept
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            pixel_array = pixel_array * slope + intercept

            volume[i, :, :] = pixel_array

        return volume


class RTStructLoader:
    """RTSTRUCT 加载器"""

    @staticmethod
    def load_rtstruct(rtstruct_path: str) -> Optional[Dict]:
        """
        加载 RTSTRUCT 文件
        返回: {roi_name: {'number': int, 'color': tuple, 'contours': list}}
        """
        try:
            ds = pydicom.dcmread(rtstruct_path)

            if not hasattr(ds, 'ROIContourSequence'):
                return None

            rois = {}

            # 获取 ROI 名称
            roi_names = {}
            if hasattr(ds, 'StructureSetROISequence'):
                for roi_info in ds.StructureSetROISequence:
                    roi_number = int(roi_info.ROINumber)
                    roi_name = str(roi_info.ROIName)
                    roi_names[roi_number] = roi_name

            # 获取 ROI 轮廓
            for roi_contour in ds.ROIContourSequence:
                roi_number = int(roi_contour.ReferencedROINumber)
                roi_name = roi_names.get(roi_number, f"ROI_{roi_number}")

                # 获取颜色
                color = (255, 0, 0)  # 默认红色
                if hasattr(roi_contour, 'ROIDisplayColor'):
                    color = tuple(int(x) for x in roi_contour.ROIDisplayColor)

                # 获取轮廓数据
                contours_3d = []
                if hasattr(roi_contour, 'ContourSequence'):
                    for contour in roi_contour.ContourSequence:
                        if hasattr(contour, 'ContourData'):
                            # ContourData 是 [x1,y1,z1, x2,y2,z2, ...]
                            data = contour.ContourData
                            points = []
                            for i in range(0, len(data), 3):
                                points.append([float(data[i]), float(data[i + 1]), float(data[i + 2])])
                            contours_3d.append(np.array(points))

                rois[roi_name] = {
                    'number': roi_number,
                    'color': color,
                    'contours_3d': contours_3d
                }

            return rois

        except Exception as e:
            print(f"加载 RTSTRUCT 失败: {e}")
            traceback.print_exc()
            return None


class DicomViewer(QMainWindow):
    """DICOM 浏览器主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM 浏览器")
        self.setGeometry(50, 50, 1600, 1000)  # 增大默认窗口尺寸

        # 数据
        self.volume = None  # 3D numpy array (Z, Y, X)
        self.metadata = None
        self.dicom_slices = []
        self.rois = {}  # {roi_name: roi_data}
        self.roi_visibility = {}  # {roi_name: bool}
        self.roi_filled = {}  # {roi_name: bool}
        self.roi_opacity = {}  # {roi_name: float}
        self.roi_line_width = 1.0

        # 显示参数
        self.current_slice_index = 0
        self.window_center = 40
        self.window_width = 400

        self.initUI()

    def initUI(self):
        """初始化 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧面板
        left_panel = self.createLeftPanel()

        # 中间图像视图
        center_panel = self.createCenterPanel()

        # 右侧属性面板
        right_panel = self.createRightPanel()

        # 使用 Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter)

    def createLeftPanel(self) -> QWidget:
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 系列列表
        series_group = QGroupBox("DICOM 系列")
        series_layout = QVBoxLayout()

        load_series_btn = QPushButton("加载 DICOM 系列")
        load_series_btn.clicked.connect(self.loadDicomSeries)
        series_layout.addWidget(load_series_btn)

        self.series_info_label = QLabel("未加载")
        series_layout.addWidget(self.series_info_label)

        series_group.setLayout(series_layout)
        layout.addWidget(series_group)

        # ROI 列表
        roi_group = QGroupBox("ROI 列表")
        roi_layout = QVBoxLayout()

        load_rtstruct_btn = QPushButton("加载 RTSTRUCT")
        load_rtstruct_btn.clicked.connect(self.loadRTStruct)
        roi_layout.addWidget(load_rtstruct_btn)

        self.roi_list = QListWidget()
        self.roi_list.itemChanged.connect(self.onROIItemChanged)
        self.roi_list.currentItemChanged.connect(self.onROISelectionChanged)
        roi_layout.addWidget(self.roi_list)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        layout.addStretch()
        return panel

    def createCenterPanel(self) -> QWidget:
        """创建中间面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 图像视图
        self.image_view = DicomImageView()
        layout.addWidget(self.image_view, 1)

        # 切片控制
        slice_control = QHBoxLayout()

        slice_control.addWidget(QLabel("切片:"))

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.onSliceChanged)
        slice_control.addWidget(self.slice_slider, 1)

        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setMinimum(0)
        self.slice_spinbox.setMaximum(0)
        self.slice_spinbox.valueChanged.connect(self.slice_slider.setValue)
        slice_control.addWidget(self.slice_spinbox)

        layout.addLayout(slice_control)

        # 按钮
        btn_layout = QHBoxLayout()

        reset_view_btn = QPushButton("重置视图")
        reset_view_btn.clicked.connect(self.image_view.resetView)
        btn_layout.addWidget(reset_view_btn)

        export_mask_btn = QPushButton("导出当前 ROI Mask")
        export_mask_btn.clicked.connect(self.exportCurrentMask)
        btn_layout.addWidget(export_mask_btn)

        export_all_btn = QPushButton("导出所有 ROI Masks")
        export_all_btn.clicked.connect(self.exportAllMasks)
        btn_layout.addWidget(export_all_btn)

        layout.addLayout(btn_layout)

        return panel

    def createRightPanel(self) -> QWidget:
        """创建右侧属性面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 显示质量设置
        quality_group = QGroupBox("显示质量")
        quality_layout = QVBoxLayout()

        # 插值模式
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("插值模式:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["高质量（平滑）", "快速（最近邻）"])
        self.interp_combo.setCurrentIndex(0)
        self.interp_combo.currentIndexChanged.connect(self.onInterpolationChanged)
        interp_layout.addWidget(self.interp_combo)
        quality_layout.addLayout(interp_layout)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # 窗宽窗位
        ww_wl_group = QGroupBox("窗宽/窗位")
        ww_wl_layout = QVBoxLayout()

        # 窗位
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("窗位:"))
        self.wl_spinbox = QSpinBox()
        self.wl_spinbox.setRange(-1024, 3071)
        self.wl_spinbox.setValue(self.window_center)
        self.wl_spinbox.valueChanged.connect(self.onWindowLevelChanged)
        wl_layout.addWidget(self.wl_spinbox)
        ww_wl_layout.addLayout(wl_layout)

        # 窗宽
        ww_layout = QHBoxLayout()
        ww_layout.addWidget(QLabel("窗宽:"))
        self.ww_spinbox = QSpinBox()
        self.ww_spinbox.setRange(1, 4096)
        self.ww_spinbox.setValue(self.window_width)
        self.ww_spinbox.valueChanged.connect(self.onWindowWidthChanged)
        ww_layout.addWidget(self.ww_spinbox)
        ww_wl_layout.addLayout(ww_layout)

        # 预设按钮
        preset_layout = QHBoxLayout()

        lung_btn = QPushButton("肺窗")
        lung_btn.clicked.connect(lambda: self.setWindowPreset(-600, 1500))
        preset_layout.addWidget(lung_btn)

        bone_btn = QPushButton("骨窗")
        bone_btn.clicked.connect(lambda: self.setWindowPreset(400, 1800))
        preset_layout.addWidget(bone_btn)

        soft_btn = QPushButton("软组织")
        soft_btn.clicked.connect(lambda: self.setWindowPreset(40, 400))
        preset_layout.addWidget(soft_btn)

        ww_wl_layout.addLayout(preset_layout)

        ww_wl_group.setLayout(ww_wl_layout)
        layout.addWidget(ww_wl_group)

        # 显示质量设置
        quality_group = QGroupBox("显示质量")
        quality_layout = QVBoxLayout()

        # 插值模式
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("插值:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["高质量", "快速"])
        self.interp_combo.setCurrentIndex(0)
        self.interp_combo.currentIndexChanged.connect(self.onInterpolationChanged)
        interp_layout.addWidget(self.interp_combo)
        quality_layout.addLayout(interp_layout)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # ROI 属性
        roi_prop_group = QGroupBox("当前 ROI 属性")
        roi_prop_layout = QVBoxLayout()

        self.roi_name_label = QLabel("无选中 ROI")
        roi_prop_layout.addWidget(self.roi_name_label)

        # 透明度
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("透明度:"))
        self.opacity_spinbox = QDoubleSpinBox()
        self.opacity_spinbox.setRange(0.0, 1.0)
        self.opacity_spinbox.setSingleStep(0.1)
        self.opacity_spinbox.setValue(0.5)
        self.opacity_spinbox.valueChanged.connect(self.onROIOpacityChanged)
        opacity_layout.addWidget(self.opacity_spinbox)
        roi_prop_layout.addLayout(opacity_layout)

        # 填充
        self.filled_checkbox = QCheckBox("填充显示")
        self.filled_checkbox.setChecked(False)
        self.filled_checkbox.stateChanged.connect(self.onROIFilledChanged)
        roi_prop_layout.addWidget(self.filled_checkbox)

        # 线条粗细
        line_width_layout = QHBoxLayout()
        line_width_layout.addWidget(QLabel("线条粗细:"))
        self.line_width_spinbox = QDoubleSpinBox()
        self.line_width_spinbox.setRange(0.5, 10.0)
        self.line_width_spinbox.setSingleStep(0.5)
        self.line_width_spinbox.setValue(1.0)
        self.line_width_spinbox.valueChanged.connect(self.onROILineWidthChanged)
        line_width_layout.addWidget(self.line_width_spinbox)
        roi_prop_layout.addLayout(line_width_layout)

        # 颜色
        color_btn = QPushButton("更改颜色")
        color_btn.clicked.connect(self.changeROIColor)
        roi_prop_layout.addWidget(color_btn)

        roi_prop_group.setLayout(roi_prop_layout)
        layout.addWidget(roi_prop_group)

        # 信息
        info_group = QGroupBox("图像信息")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("无数据")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        return panel

    def onInterpolationChanged(self, index):
        """插值模式改变"""
        # 0: 平滑, 1: 最近邻
        self.image_view.use_smooth_transform = (index == 0)
        self.updateDisplay()

    def loadDicomSeries(self):
        """加载 DICOM 系列"""
        folder = QFileDialog.getExistingDirectory(self, "选择 DICOM 系列文件夹")
        if not folder:
            return

        # 显示进度对话框
        progress = QProgressDialog("加载 DICOM 系列中...", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        try:
            self.volume, self.metadata, self.dicom_slices = DicomSeriesLoader.load_dicom_series(folder)

            if self.volume is None:
                QMessageBox.warning(self, "错误", "无法加载 DICOM 系列")
                return

            # 更新 UI
            self.current_slice_index = self.metadata['num_slices'] // 2
            self.slice_slider.setMaximum(self.metadata['num_slices'] - 1)
            self.slice_slider.setValue(self.current_slice_index)
            self.slice_spinbox.setMaximum(self.metadata['num_slices'] - 1)
            self.slice_spinbox.setValue(self.current_slice_index)

            self.series_info_label.setText(
                f"已加载: {self.metadata['num_slices']} 切片\n"
                f"尺寸: {self.metadata['rows']}x{self.metadata['cols']}\n"
                f"间距: {self.metadata['spacing']}"
            )

            self.updateImageInfo()
            self.updateDisplay()

            QMessageBox.information(self, "成功", f"成功加载 {self.metadata['num_slices']} 个切片")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {str(e)}\n{traceback.format_exc()}")
        finally:
            progress.close()

    def loadRTStruct(self):
        """加载 RTSTRUCT"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择 RTSTRUCT 文件", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        if not filepath:
            return

        try:
            self.rois = RTStructLoader.load_rtstruct(filepath)

            if not self.rois:
                QMessageBox.warning(self, "警告", "未找到 ROI 数据")
                return

            # 更新 ROI 列表
            self.roi_list.clear()
            for roi_name, roi_data in self.rois.items():
                item = QListWidgetItem(roi_name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)

                # 设置颜色显示
                color = QColor(*roi_data['color'])
                item.setForeground(color)

                self.roi_list.addItem(item)

                # 初始化 ROI 状态
                self.roi_visibility[roi_name] = True
                self.roi_filled[roi_name] = False
                self.roi_opacity[roi_name] = 0.5
                self.roi_line_width = 1.0

            self.updateDisplay()

            QMessageBox.information(self, "成功", f"成功加载 {len(self.rois)} 个 ROI")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载 RTSTRUCT 失败: {str(e)}\n{traceback.format_exc()}")

    def onSliceChanged(self, value):
        """切片改变"""
        self.current_slice_index = value
        self.slice_spinbox.setValue(value)
        self.updateDisplay()
        self.updateImageInfo()

    def onWindowLevelChanged(self, value):
        """窗位改变"""
        self.window_center = value
        self.updateDisplay()

    def onWindowWidthChanged(self, value):
        """窗宽改变"""
        self.window_width = value
        self.updateDisplay()

    def setWindowPreset(self, center, width):
        """设置窗宽窗位预设"""
        self.window_center = center
        self.window_width = width
        self.wl_spinbox.setValue(center)
        self.ww_spinbox.setValue(width)
        self.updateDisplay()

    def onROISelectionChanged(self, current, previous):
        """ROI 选择改变时更新属性面板"""
        self.line_width_spinbox.setValue(self.roi_line_width)
        if current:
            roi_name = current.text()
            # 更新属性控件的值
            self.roi_name_label.setText(f"当前 ROI: {roi_name}")
            self.opacity_spinbox.setValue(self.roi_opacity.get(roi_name, 0.5))
            self.filled_checkbox.setChecked(self.roi_filled.get(roi_name, False))
        else:
            self.roi_name_label.setText("无选中 ROI")

    def onROIItemChanged(self, item):
        """ROI 项改变（勾选/取消）"""
        roi_name = item.text()
        is_checked = item.checkState() == Qt.Checked
        self.roi_visibility[roi_name] = is_checked
        self.updateDisplay()

    def onROIOpacityChanged(self, value):
        """ROI 透明度改变"""
        current_item = self.roi_list.currentItem()
        if current_item:
            roi_name = current_item.text()
            self.roi_opacity[roi_name] = value
            self.updateDisplay()

    def onROIFilledChanged(self, state):
        """ROI 填充状态改变"""
        current_item = self.roi_list.currentItem()
        if current_item:
            roi_name = current_item.text()
            self.roi_filled[roi_name] = (state == Qt.Checked)
            self.updateDisplay()

    def onROILineWidthChanged(self, value):
        """ROI 线条粗细改变"""
        self.roi_line_width = value
        current_item = self.roi_list.currentItem()
        if current_item:
            roi_name = current_item.text()
            # 更新图像视图的线条粗细
            self.image_view.contour_line_width = value
            self.updateDisplay()

    def changeROIColor(self):
        """更改 ROI 颜色"""
        current_item = self.roi_list.currentItem()
        if not current_item:
            return

        roi_name = current_item.text()
        current_color = QColor(*self.rois[roi_name]['color'])

        color = QColorDialog.getColor(current_color, self, "选择 ROI 颜色")
        if color.isValid():
            self.rois[roi_name]['color'] = (color.red(), color.green(), color.blue())
            current_item.setForeground(color)
            self.updateDisplay()

    def updateDisplay(self):
        """更新图像显示"""
        if self.volume is None:
            return

        # 获取当前切片
        slice_data = self.volume[self.current_slice_index, :, :]

        # 应用窗宽窗位
        windowed_image = self.applyWindowLevel(slice_data, self.window_center, self.window_width)

        # 准备 ROI 数据
        rois_to_draw = self.prepareROIsForSlice(self.current_slice_index)

        # 更新显示
        self.image_view.setImage(windowed_image, rois_to_draw)

    def applyWindowLevel(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """应用窗宽窗位，返回 8-bit 图像"""
        min_val = center - width / 2
        max_val = center + width / 2

        windowed = np.clip(image, min_val, max_val)
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        return windowed

    def prepareROIsForSlice(self, slice_index: int) -> List[Dict]:
        """准备当前切片的 ROI 数据"""
        if not self.metadata or not self.rois:
            return []

        rois_to_draw = []

        # 计算当前切片的 Z 坐标
        slice_z = self.metadata['origin'][2] + slice_index * self.metadata['spacing'][0]
        tolerance = self.metadata['spacing'][0] / 2  # 容差

        for roi_name, roi_data in self.rois.items():
            if not self.roi_visibility.get(roi_name, True):
                continue

            contours_2d = []

            # 遍历所有 3D 轮廓
            for contour_3d in roi_data['contours_3d']:
                if len(contour_3d) == 0:
                    continue

                # 检查轮廓是否在当前切片
                contour_z = contour_3d[0, 2]
                if abs(contour_z - slice_z) < tolerance:
                    # 转换物理坐标到像素坐标
                    contour_2d = self.worldToPixel(contour_3d[:, :2])
                    contours_2d.append(contour_2d)

            if contours_2d:
                rois_to_draw.append({
                    'contours': contours_2d,
                    'color': roi_data['color'],
                    'filled': self.roi_filled.get(roi_name, False),
                    'opacity': self.roi_opacity.get(roi_name, 0.5),
                    'line_width': self.roi_line_width
                })
        return rois_to_draw

    def worldToPixel(self, world_coords: np.ndarray) -> np.ndarray:
        """将物理坐标转换为像素坐标
        
        Args:
            world_coords: Nx2 array of (x, y) world coordinates
        
        Returns:
            Nx2 array of (x, y) pixel coordinates
        """
        if self.metadata is None:
            return world_coords

        # 获取第一个切片的信息
        origin = self.metadata['origin']  # [x, y, z]
        spacing = self.metadata['spacing']  # [z, y, x] - 注意顺序

        # world_coords 是 [x, y] 坐标
        # 转换：像素坐标 = (世界坐标 - 原点) / 间距
        pixel_x = (world_coords[:, 0] - origin[0]) / spacing[2]  # x spacing
        pixel_y = (world_coords[:, 1] - origin[1]) / spacing[1]  # y spacing

        pixel_coords = np.column_stack([pixel_x, pixel_y])

        return pixel_coords

    def updateImageInfo(self):
        """更新图像信息"""
        if self.volume is None:
            self.info_label.setText("无数据")
            return

        slice_data = self.volume[self.current_slice_index, :, :]

        info_text = (
            f"切片: {self.current_slice_index + 1}/{self.metadata['num_slices']}\n"
            f"尺寸: {slice_data.shape}\n"
            f"值范围: [{slice_data.min():.1f}, {slice_data.max():.1f}]\n"
            f"窗位/窗宽: {self.window_center}/{self.window_width}"
        )

        self.info_label.setText(info_text)

    def exportCurrentMask(self):
        """导出当前选中 ROI 的二值 mask"""
        current_item = self.roi_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个 ROI")
            return

        roi_name = current_item.text()
        self.exportROIMask(roi_name)

    def exportAllMasks(self):
        """导出所有 ROI 的二值 mask"""
        if not self.rois:
            QMessageBox.warning(self, "警告", "没有可导出的 ROI")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not folder:
            return

        for roi_name in self.rois.keys():
            output_path = os.path.join(folder, f"{roi_name}_mask.nii.gz")
            self.exportROIMask(roi_name, output_path)

        QMessageBox.information(self, "成功", f"成功导出 {len(self.rois)} 个 ROI masks")

    def exportROIMask(self, roi_name: str, output_path: str = None):
        """导出单个 ROI 的二值 mask"""
        if self.volume is None or roi_name not in self.rois:
            return

        if not SKIMAGE_AVAILABLE:
            QMessageBox.warning(self, "警告", "需要安装 scikit-image 才能导出 mask\n请运行: pip install scikit-image")
            return

        if output_path is None:
            output_path, _ = QFileDialog.getSaveFileName(
                self, "保存 Mask", f"{roi_name}_mask.nii.gz", "NIfTI Files (*.nii.gz)"
            )
            if not output_path:
                return

        try:
            # 创建二值 mask
            mask = np.zeros_like(self.volume, dtype=np.uint8)

            roi_data = self.rois[roi_name]

            # 遍历所有轮廓
            for contour_3d in roi_data['contours_3d']:
                if len(contour_3d) == 0:
                    continue

                # 找到对应的切片
                contour_z = contour_3d[0, 2]
                origin_z = self.metadata['origin'][2]
                spacing_z = self.metadata['spacing'][0]
                slice_index = int(round((contour_z - origin_z) / spacing_z))

                if 0 <= slice_index < self.metadata['num_slices']:
                    # 转换为像素坐标
                    contour_2d = self.worldToPixel(contour_3d[:, :2])

                    # 填充轮廓
                    # contour_2d 是 [x, y] 格式，polygon 需要 (row, col) = (y, x)
                    try:
                        rr, cc = polygon(contour_2d[:, 1], contour_2d[:, 0], mask.shape[1:])
                        # 确保索引在范围内
                        valid_idx = (rr >= 0) & (rr < mask.shape[1]) & (cc >= 0) & (cc < mask.shape[2])
                        rr = rr[valid_idx]
                        cc = cc[valid_idx]
                        mask[slice_index, rr, cc] = 1
                    except Exception as e:
                        print(f"填充轮廓失败 (切片 {slice_index}): {e}")
                        continue

            # 保存为 NIfTI
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.SetSpacing(self.metadata['spacing'][::-1])  # [z,y,x] -> [x,y,z]
            mask_sitk.SetOrigin(self.metadata['origin'])

            # 设置方向矩阵
            direction_matrix = self.metadata['direction']
            if direction_matrix.shape == (3, 3):
                mask_sitk.SetDirection(direction_matrix.flatten())

            sitk.WriteImage(mask_sitk, output_path)

            if output_path.endswith('.nii.gz') or output_path.endswith('.nii'):
                QMessageBox.information(self, "成功", f"Mask 已保存到:\n{output_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}\n{traceback.format_exc()}")


def main():
    # 启用高 DPI 支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 设置应用样式（可选）
    app.setStyle('Fusion')  # 使用 Fusion 样式以获得更好的跨平台外观

    viewer = DicomViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
