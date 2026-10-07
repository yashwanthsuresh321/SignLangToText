import cv2
import time

from src.ui.layout import LayoutManager
from src.ui.theme import Theme
from src.ui.constants import Constants

from src.ui.components.header import HeaderComponent
from src.ui.components.dataset import DatasetComponent
from src.ui.components.prediction import PredictionComponent
from src.ui.components.sentence import SentenceComponent
from src.ui.components.dynamic import DynamicPanelComponent
from src.ui.components.separator import SeparatorComponent
from src.ui.components.progress_bar import ProgressBarComponent

class UI:
    """Main UI Coordinator."""
    
    def __init__(self):
        self.previous_time = time.time()
        self.layout = LayoutManager()
        
        # Instantiate Components
        self.header_comp = HeaderComponent()
        self.dataset_comp = DatasetComponent()
        self.prediction_comp = PredictionComponent()
        self.sentence_comp = SentenceComponent()
        self.dynamic_comp = DynamicPanelComponent()
        self.separator_comp = SeparatorComponent()
        self.progress_bar_comp = ProgressBarComponent()

    def calculate_fps(self) -> int:
        current_time = time.time()
        fps = 1 / max(current_time - self.previous_time, 0.0001)
        self.previous_time = current_time
        return int(fps)

    def draw_overlay(
        self,
        frame,
        hand_count=0,
        handedness="-",
        confidence=0.0,
        resolution="",
        prediction="-",
        prediction_confidence=0.0,
        sentence="",
        hold_progress=0.0,
        letter_added=False,
        last_added_letter="",
        current_label="-",
        current_count=0,
        target_count=300,
        progress=0,
        status="Ready",
        session_count=0,
        dynamic_label='-',
        dynamic_frames=0,
        dynamic_target=20,
        dynamic_recording=False,
        dynamic_collected=0,
        dynamic_goal=150,
        dynamic_j_count=0,
        dynamic_z_count=0
    ):
        """
        Main entry point for drawing the UI.
        Matches the signature of the previous implementation exactly.
        """
        h, w = frame.shape[:2]
        
        # 1. Update Layout
        self.layout.calculate_layout(frame_width=w, frame_height=h)
        
        # Background removal: Camera feed now occupies 100% of background.
        # Floating glass cards provide their own backgrounds natively.

        # 3. FPS
        fps = self.calculate_fps()

        # 4. Header
        header_data = {
            'fps': fps,
            'hand_count': hand_count,
            'handedness': handedness,
            'confidence': confidence,
            'resolution': resolution
        }
        self.header_comp.draw(frame, self.layout.get_rect('header'), header_data)

        # 5. Dataset Panel
        dataset_data = {
            'current_label': current_label,
            'current_count': current_count,
            'target_count': target_count,
            'session_count': session_count,
            'status': status,
            'progress': progress,
            'progress_bar_comp': self.progress_bar_comp
        }
        self.dataset_comp.draw(frame, self.layout.get_rect('dataset'), dataset_data)

        # 6. Prediction Panel
        prediction_data = {
            'prediction': prediction,
            'confidence': prediction_confidence,
            'hold_progress': hold_progress
        }
        self.prediction_comp.draw(frame, self.layout.get_rect('prediction'), prediction_data)

        # 7. Sentence Panel
        sentence_data = {
            'sentence': sentence,
            'letter_added': letter_added,
            'last_added_letter': last_added_letter
        }
        self.sentence_comp.draw(frame, self.layout.get_rect('sentence'), sentence_data)

        # 8. Dynamic Recording Panel
        dynamic_data = {
            'label': dynamic_label,
            'frames': dynamic_frames,
            'target': dynamic_target,
            'recording': dynamic_recording,
            'collected': dynamic_collected,
            'goal': dynamic_goal,
            'j_count': dynamic_j_count,
            'z_count': dynamic_z_count
        }
        self.dynamic_comp.draw(frame, self.layout.get_rect('dynamic'), dynamic_data)

        # 9. Separators (Obsolete)
        # Separators are no longer drawn in the new dashboard layout.

        return frame
