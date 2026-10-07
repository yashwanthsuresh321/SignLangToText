class Constants:
    """
    Professional Dashboard UI Constants

    These values control ONLY the geometry of the UI.

    Layout philosophy:
    - Webcam remains the primary focus.
    - Floating cards occupy minimal space.
    - Consistent spacing and padding.
    - Responsive proportions.
    """

    # ==========================================================
    # SCREEN MARGINS
    # ==========================================================

    # Distance from screen edges
    MARGIN_X = 24
    MARGIN_Y = 20

    # Gap between floating cards
    GAP = 18

    # Internal padding inside cards
    PADDING = 20

    # ==========================================================
    # PANEL HEIGHTS
    # ==========================================================

    # Compact floating header
    HEADER_HEIGHT = 52

    # Dataset / Prediction / Sentence row
    DATASET_PRED_SENT_HEIGHT = 120

    # Dynamic recognition panel
    DYNAMIC_PANEL_HEIGHT = 95

    # ==========================================================
    # PROGRESS BARS
    # ==========================================================

    PROGRESS_BAR_HEIGHT = 20
    DYNAMIC_PROGRESS_BAR_HEIGHT = 12

    # ==========================================================
    # WIDTH DISTRIBUTION
    #
    # Dataset | Prediction | Sentence
    #
    # Total = 1.00
    # ==========================================================

    PREDICTION_WIDTH_PCT = 0.30
    SENTENCE_WIDTH_PCT = 0.70

    # ==========================================================
    # CARD RADIUS
    # ==========================================================

    CARD_RADIUS = 18

    # ==========================================================
    # CARD BORDER
    # ==========================================================

    BORDER_THICKNESS = 1

    # ==========================================================
    # ICON SIZES
    # ==========================================================

    SMALL_ICON = 16
    MEDIUM_ICON = 22
    LARGE_ICON = 30

    # ==========================================================
    # TEXT SPACING
    # ==========================================================

    TITLE_MARGIN_TOP = 16
    SECTION_SPACING = 14
    LINE_SPACING = 28

    # ==========================================================
    # PREDICTION PANEL
    # ==========================================================

    PREDICTION_LETTER_SCALE = 2.8
    PREDICTION_CONFIDENCE_SCALE = 0.75
    PREDICTION_SOURCE_SCALE = 0.60

    # ==========================================================
    # SENTENCE PANEL
    # ==========================================================

    MAX_SENTENCE_LENGTH = 42

    # ==========================================================
    # DATASET PANEL
    # ==========================================================

    DATASET_PROGRESS_HEIGHT = 18

    # ==========================================================
    # DYNAMIC PANEL
    # ==========================================================

    DYNAMIC_STATUS_RADIUS = 8

    # ==========================================================
    # ANIMATION
    # ==========================================================

    FADE_SPEED = 0.12
    PULSE_SPEED = 0.15

    # ==========================================================
    # RESPONSIVE BREAKPOINTS
    # ==========================================================

    SMALL_SCREEN_WIDTH = 960
    LARGE_SCREEN_WIDTH = 1600