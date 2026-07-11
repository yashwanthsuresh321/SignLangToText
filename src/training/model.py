"""
SignLanguageAI
Version 0.5.0

MLP Model
"""

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)


class SignLanguageModel:

    def __init__(
        self,
        input_shape,
        num_classes
    ):

        self.input_shape = input_shape

        self.num_classes = num_classes

        self.model = None

    # ==========================================================
    # Build Model
    # ==========================================================

    def build(self):

        self.model = Sequential([

            Input(
                shape=(self.input_shape,)
            ),

            # -----------------------------------------
            # Hidden Layer 1
            # -----------------------------------------

            Dense(
                128,
                activation="relu"
            ),

            BatchNormalization(),

            Dropout(
                0.30
            ),

            # -----------------------------------------
            # Hidden Layer 2
            # -----------------------------------------

            Dense(
                64,
                activation="relu"
            ),

            BatchNormalization(),

            Dropout(
                0.20
            ),

            # -----------------------------------------
            # Hidden Layer 3
            # -----------------------------------------

            Dense(
                32,
                activation="relu"
            ),

            Dropout(
                0.15
            ),

            # -----------------------------------------
            # Output Layer
            # -----------------------------------------

            Dense(
                self.num_classes,
                activation="softmax"
            )

        ])

        return self.model

    # ==========================================================
    # Compile Model
    # ==========================================================

    def compile(

        self,

        learning_rate=0.001

    ):

        self.model.compile(

            optimizer=tf.keras.optimizers.Adam(

                learning_rate=learning_rate

            ),

            loss="sparse_categorical_crossentropy",

            metrics=[

                "accuracy"

            ]

        )

    # ==========================================================
    # Summary
    # ==========================================================

    def summary(self):

        print()

        print("=" * 60)

        print("        SIGNLANGUAGEAI MODEL")

        print("=" * 60)

        self.model.summary()

        print("=" * 60)

    # ==========================================================
    # Get Model
    # ==========================================================

    def get_model(self):

        return self.model