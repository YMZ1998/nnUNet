import os
import sys
import time
import json
from os.path import join, exists, dirname, abspath
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                             QFileDialog, QTextEdit, QSpinBox, QCheckBox,
                             QComboBox, QGroupBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import torch

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


class PredictWorker(QThread):
    """预测工作线程"""
    finished = pyqtSignal(str, float)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    
    def __init__(self, model_dir, checkpoint_name, input_file, output_file, 
                 device_id, tile_step_size, use_gaussian, use_mirroring):
        super().__init__()
        self.model_dir = model_dir
        self.checkpoint_name = checkpoint_name
        self.input_file = input_file
        self.output_file = output_file
        self.device_id = device_id
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        
    def run(self):
        try:
            start_time = time.time()
            
            self.log.emit(f"初始化预测器...")
            self.log.emit(f"模型目录: {self.model_dir}")
            self.log.emit(f"检查点: {self.checkpoint_name}")
            self.log.emit(f"设备: cuda:{self.device_id}")
            
            predictor = nnUNetPredictor(
                tile_step_size=self.tile_step_size,
                use_gaussian=self.use_gaussian,
                use_mirroring=self.use_mirroring,
                perform_everything_on_device=True,
                device=torch.device('cuda', self.device_id),
                verbose=True,
                verbose_preprocessing=True,
                allow_tqdm=True
            )
            
            self.log.emit("加载模型...")
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=self.model_dir,
                use_folds="all",
                checkpoint_name=self.checkpoint_name,
            )
            
            self.log.emit(f"读取输入图像: {self.input_file}")
            img, props = SimpleITKIO().read_images([self.input_file])
            self.log.emit(f"图像属性: {props}")
            
            self.log.emit("开始预测...")
            pred = predictor.predict_single_npy_array(img, props, None, None, False)
            self.log.emit(f"预测形状: {pred.shape}")
            
            self.log.emit(f"保存结果到: {self.output_file}")
            SimpleITKIO().write_seg(pred, self.output_file, props)
            self.log.emit("保存完成")
            print(f"Prediction saved to: {self.output_file}")
            
            elapsed_time = time.time() - start_time
            self.finished.emit(self.output_file, elapsed_time)
            
        except Exception as e:
            self.error.emit(str(e))


class nnUNetPredictorGUI(QMainWindow):
    """nnU-Net单文件预测GUI"""
    
    CONFIG_FILE = "prediction_config.json"
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        self.setWindowTitle("nnU-Net 单文件预测器")
        self.setGeometry(100, 100, 900, 700)
        
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout()
        
        # 任务ID
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("任务ID:"))
        self.task_id_spin = QSpinBox()
        self.task_id_spin.setRange(1, 9999)
        self.task_id_spin.setValue(999)
        self.task_id_spin.valueChanged.connect(self.update_model_dir)
        task_layout.addWidget(self.task_id_spin)
        task_layout.addStretch()
        model_layout.addLayout(task_layout)
        
        # 模型配置
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("模型配置:"))
        self.model_config_combo = QComboBox()
        self.model_config_combo.addItems([
            'nnUNetTrainer__nnUNetPlans__3d_fullres',
            'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres',
            'nnUNetTrainer__nnUNetPlans__2d',
            'nnUNetTrainer__nnUNetPlans__3d_lowres',
            'nnUNetTrainer__nnUNetPlans__3d_cascade_fullres'
        ])
        self.model_config_combo.currentTextChanged.connect(self.update_model_dir)
        config_layout.addWidget(self.model_config_combo)
        model_layout.addLayout(config_layout)
        
        # 检查点
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(QLabel("检查点:"))
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.addItems([
            'checkpoint_best.pth',
            'checkpoint_final.pth'
        ])
        checkpoint_layout.addWidget(self.checkpoint_combo)
        checkpoint_layout.addStretch()
        model_layout.addLayout(checkpoint_layout)
        
        # 模型目录
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("模型目录:"))
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.model_dir_edit)
        model_layout.addLayout(dir_layout)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 输入输出组
        io_group = QGroupBox("输入/输出")
        io_layout = QVBoxLayout()
        
        # 输入文件
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入文件:"))
        self.input_file_edit = QLineEdit()
        input_layout.addWidget(self.input_file_edit)
        self.input_browse_btn = QPushButton("浏览...")
        self.input_browse_btn.clicked.connect(self.browse_input_file)
        input_layout.addWidget(self.input_browse_btn)
        io_layout.addLayout(input_layout)
        
        # 输出文件
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出文件:"))
        self.output_file_edit = QLineEdit()
        output_layout.addWidget(self.output_file_edit)
        self.output_browse_btn = QPushButton("浏览...")
        self.output_browse_btn.clicked.connect(self.browse_output_file)
        output_layout.addWidget(self.output_browse_btn)
        io_layout.addLayout(output_layout)
        
        io_group.setLayout(io_layout)
        main_layout.addWidget(io_group)
        
        # 预测参数组
        param_group = QGroupBox("预测参数")
        param_layout = QVBoxLayout()
        
        # GPU设备
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("GPU设备ID:"))
        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 7)
        self.device_spin.setValue(0)
        device_layout.addWidget(self.device_spin)
        device_layout.addStretch()
        param_layout.addLayout(device_layout)
        
        # Tile步长
        tile_layout = QHBoxLayout()
        tile_layout.addWidget(QLabel("Tile步长:"))
        self.tile_step_spin = QSpinBox()
        self.tile_step_spin.setRange(0, 10)
        self.tile_step_spin.setValue(5)
        tile_layout.addWidget(self.tile_step_spin)
        tile_layout.addWidget(QLabel("(实际值 × 0.1)"))
        tile_layout.addStretch()
        param_layout.addLayout(tile_layout)
        
        # 高斯权重
        self.gaussian_check = QCheckBox("使用高斯权重")
        self.gaussian_check.setChecked(True)
        param_layout.addWidget(self.gaussian_check)
        
        # 镜像增强
        self.mirroring_check = QCheckBox("使用镜像增强")
        self.mirroring_check.setChecked(False)
        param_layout.addWidget(self.mirroring_check)
        
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 日志输出
        log_label = QLabel("运行日志:")
        main_layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        main_layout.addWidget(self.log_text)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        button_layout.addWidget(self.predict_btn)
        
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        button_layout.addWidget(self.clear_log_btn)
        
        self.open_output_btn = QPushButton("打开输出目录")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_output_btn.setEnabled(False)
        button_layout.addWidget(self.open_output_btn)
        
        main_layout.addLayout(button_layout)
        
        # 初始化模型目录
        self.update_model_dir()
        
    def update_model_dir(self):
        """更新模型目录"""
        task_id = self.task_id_spin.value()
        dataset_name = maybe_convert_to_dataset_name(task_id)
        model_config = self.model_config_combo.currentText()
        model_dir = join(nnUNet_results, dataset_name, model_config)
        self.model_dir_edit.setText(model_dir)
        
    def browse_input_file(self):
        """浏览输入文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择输入文件", "", "NIfTI文件 (*.nii.gz *.nii);;所有文件 (*.*)"
        )
        if file_path:
            self.input_file_edit.setText(file_path)
            # 自动设置输出文件
            output_path = file_path.replace('.nii.gz', '_seg.nii.gz')
            self.output_file_edit.setText(output_path)
            
    def browse_output_file(self):
        """浏览输出文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件", "", "NIfTI文件 (*.nii.gz *.nii);;所有文件 (*.*)"
        )
        if file_path:
            self.output_file_edit.setText(file_path)
            
    def start_prediction(self):
        """开始预测"""
        # 验证输入
        input_file = self.input_file_edit.text().strip()
        output_file = self.output_file_edit.text().strip()
        model_dir = self.model_dir_edit.text().strip()
        
        if not input_file or not exists(input_file):
            self.log_message("错误: 请选择有效的输入文件", "red")
            return
            
        if not output_file:
            self.log_message("错误: 请指定输出文件路径", "red")
            return
            
        if not exists(model_dir):
            self.log_message(f"错误: 模型目录不存在: {model_dir}", "red")
            return
            
        # 保存当前配置
        self.save_config()
        
        # 禁用按钮
        self.predict_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.open_output_btn.setEnabled(False)
        
        # 创建工作线程
        self.worker = PredictWorker(
            model_dir=model_dir,
            checkpoint_name=self.checkpoint_combo.currentText(),
            input_file=input_file,
            output_file=output_file,
            device_id=self.device_spin.value(),
            tile_step_size=self.tile_step_spin.value() * 0.1,
            use_gaussian=self.gaussian_check.isChecked(),
            use_mirroring=self.mirroring_check.isChecked()
        )
        
        self.worker.log.connect(self.log_message)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.on_prediction_error)
        
        self.log_message("=" * 80)
        self.log_message("开始预测任务...")
        self.worker.start()
        
    def on_prediction_finished(self, output_file, elapsed_time):
        """预测完成"""
        self.log_message(f"预测完成! 用时: {elapsed_time:.2f} 秒", "green")
        self.log_message(f"结果已保存至: {output_file}", "green")
        self.log_message("=" * 80)
        
        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.open_output_btn.setEnabled(True)
        
    def on_prediction_error(self, error_msg):
        """预测出错"""
        self.log_message(f"错误: {error_msg}", "red")
        self.log_message("=" * 80)
        
        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def log_message(self, message, color="black"):
        """输出日志消息"""
        if color != "black":
            message = f'<span style="color: {color};">{message}</span>'
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def open_output_folder(self):
        """打开输出文件夹"""
        output_file = self.output_file_edit.text().strip()
        if output_file and exists(output_file):
            output_dir = dirname(output_file)
            os.startfile(output_dir)
            
    def save_config(self):
        """保存配置到JSON文件"""
        config = {
            'task_id': self.task_id_spin.value(),
            'model_config': self.model_config_combo.currentText(),
            'checkpoint': self.checkpoint_combo.currentText(),
            'input_file': self.input_file_edit.text(),
            'output_file': self.output_file_edit.text(),
            'device_id': self.device_spin.value(),
            'tile_step_size': self.tile_step_spin.value(),
            'use_gaussian': self.gaussian_check.isChecked(),
            'use_mirroring': self.mirroring_check.isChecked()
        }
        
        config_path = join(dirname(abspath(__file__)), self.CONFIG_FILE)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log_message(f"保存配置失败: {e}", "orange")
            
    def load_config(self):
        """从JSON文件加载配置"""
        config_path = join(dirname(abspath(__file__)), self.CONFIG_FILE)
        if not exists(config_path):
            return
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            self.task_id_spin.setValue(config.get('task_id', 999))
            
            model_config = config.get('model_config', '')
            if model_config:
                index = self.model_config_combo.findText(model_config)
                if index >= 0:
                    self.model_config_combo.setCurrentIndex(index)
                    
            checkpoint = config.get('checkpoint', '')
            if checkpoint:
                index = self.checkpoint_combo.findText(checkpoint)
                if index >= 0:
                    self.checkpoint_combo.setCurrentIndex(index)
                    
            self.input_file_edit.setText(config.get('input_file', ''))
            self.output_file_edit.setText(config.get('output_file', ''))
            self.device_spin.setValue(config.get('device_id', 0))
            self.tile_step_spin.setValue(config.get('tile_step_size', 5))
            self.gaussian_check.setChecked(config.get('use_gaussian', True))
            self.mirroring_check.setChecked(config.get('use_mirroring', False))
            
            self.log_message("已加载上次配置", "blue")
            
        except Exception as e:
            self.log_message(f"加载配置失败: {e}", "orange")
            

def main():
    app = QApplication(sys.argv)
    window = nnUNetPredictorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
