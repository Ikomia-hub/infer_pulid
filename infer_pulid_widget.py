from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_pulid.infer_pulid_process import InferPulidParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferPulidWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferPulidParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Number of inference steps
        self.spin_number_of_steps = pyqtutils.append_spin(
                                                    self.grid_layout,
                                                    "Number of steps",
                                                    self.parameters.num_inference_steps,
                                                    min=1, step=1
                                                    )

        # Guidance scale
        self.spin_guidance_scale = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Guidance scale",
                                                        self.parameters.guidance_scale,
                                                        min=0, step=0.1, decimals=1
                                                    )

        # ID Guidance scale
        self.spin_guidance_scale_id = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "ID Guidance scale",
                                                        self.parameters.guidance_scale,
                                                        min=0, step=0.1, decimals=1
                                                    )

        # Negative prompt
        self.edit_negative_prompt = pyqtutils.append_edit(
                                                    self.grid_layout,
                                                    "Negative prompt",
                                                    self.parameters.negative_prompt
                                                    )

        # Output width
        self.spin_width = pyqtutils.append_spin(
                                            self.grid_layout, 
                                            "Output width", 
                                            self.parameters.width, 
                                            min=256, step=8
                                            )

        # Output height
        self.spin_height = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Output height",
                                            self.parameters.height,
                                            min=256,
                                            step=8
                                            )

        # Seed
        self.spin_seed = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Seed",
                                            self.parameters.seed,
                                            min=-1, step=1
                                            )

        # Number of output images
        self.spin_num_images_per_prompt = pyqtutils.append_spin(
                                                    self.grid_layout,
                                                    "Number of outputs",
                                                    self.parameters.num_images_per_prompt,
                                                    min=1, step=1
                                                    )

        # Mode
        self.combo_mode = pyqtutils.append_combo(
            self.grid_layout, "Mode")
        self.combo_mode.addItem("fidelity")
        self.combo_mode.addItem("extremely style")

        self.combo_mode.setCurrentText(self.parameters.mode)

        # ID mix (mixing to inputs)
        self.check_id_mix = pyqtutils.append_check(self.grid_layout,
                                                 "ID Mix",
                                                 self.parameters.id_mix)

        # Set widget layout
        self.set_layout(layout_ptr)


    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.mode = self.combo_mode.currentText()
        self.parameters.num_inference_steps = self.spin_number_of_steps.value()
        self.parameters.num_images_per_prompt = self.spin_num_images_per_prompt.value()
        self.parameters.guidance_scale = self.spin_guidance_scale.value()
        self.parameters.negative_prompt = self.edit_negative_prompt.text()
        self.parameters.width = self.spin_width.value()
        self.parameters.height = self.spin_height.value()
        self.parameters.seed = self.spin_seed.value()
        self.parameters.id_mix = self.check_id_mix.isChecked()

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferPulidWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_pulid"

    def create(self, param):
        # Create widget object
        return InferPulidWidget(param, None)
