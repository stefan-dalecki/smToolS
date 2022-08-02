"""Use pre-processed data to identify protein trajectories"""

import os
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class File:
    def __init__(self) -> None:
        root = Tk()
        root.withdraw()
        self._file_path = filedialog.askopenfilename()
        self._extension = os.path.splitext(self._file_path)[1]
        self._df = None
        self.data = None

    def read_file(self):
        print(f"Reading : {self._file_path}")
        if self._extension == ".csv":
            self._df = pd.read_csv(self._file_path)
        elif self._extension == ".xlsx":
            self._df = pd.read_excel(self._file_path)
        print("   Reading Complete")
        return self

    def identify_parameters(
        self, *, parameters: list = ["Average_Brightness", "Length (frames)", "MSD"]
    ):
        assert set(parameters) & set(self._df.columns)
        self.data = self._df[["Trajectory", *parameters]]
        self.data.drop_duplicates(inplace=True)
        self.data = self.data.sort_values(by="Trajectory").reset_index(drop=True)
        return self


class Digestion:
    def __init__(self, df):
        self._df = df
        self.array = None
        self.shuffled_inputs = None
        self.shuffled_outputs = None

    def __call__(self):
        df_with_keepers = self.keep_these()
        parameter_values = df_with_keepers.iloc[:, 1:-1]
        normalized_values = self.normalize(parameter_values)
        identities = df_with_keepers.iloc[:, -1]

        self.array = df_with_keepers.to_numpy()
        inputs = normalized_values.to_numpy()
        outputs = identities.to_numpy()

        self.shuffled_inputs, self.shuffled_outputs = self.shuffle_it(
            self.array, inputs, outputs
        )

    def keep_these(
        self,
        *,
        brightness: tuple = (3.1, 3.8),
        min_length: int = 10,
        diffusion: tuple = (0.3, 3.5),
    ) -> pd.DataFrame:
        assert min_length < np.max(self._df["Length (frames)"])
        keepers = self._df.loc[
            (
                (self._df["Average_Brightness"].between(brightness[0], brightness[1]))
                & (self._df["Length (frames)"] > min_length)
                & (self._df["MSD"].between(diffusion[0], diffusion[1]))
            )
        ].assign(Keep=1)
        return pd.merge(self._df, keepers, how="left").fillna(0)

    @staticmethod
    def normalize(df) -> pd.DataFrame:
        for col in df:
            mean = np.mean(df[col])
            std = np.std(df[col])
            df[col] = (df[col] - mean) / std
            print(f"{col=}, {mean=}, {std=}")
        return df

    @staticmethod
    def shuffle_it(array, inputs, outputs):
        indeces_permutation = np.random.permutation(len(array))
        shuffled_inputs = inputs[indeces_permutation]
        shuffled_outputs = outputs[indeces_permutation]
        return shuffled_inputs, shuffled_outputs


class Learning:
    def __init__(self, array, inputs, outputs) -> None:
        self._array = array
        self._inputs = inputs
        self._outputs = outputs
        self._performance_metrics = None

    @staticmethod
    def pipeline_model():
        model = keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def prototype_model(self, *, num_epochs: int = 50) -> None:
        num_validation_samples = int(0.3 * len(self._array))
        train_inputs = self._inputs[:num_validation_samples]
        train_targets = self._outputs[:num_validation_samples]
        validation_inputs = self._inputs[num_validation_samples:]
        validation_targets = self._outputs[num_validation_samples:]
        temp_model = Learning.pipeline_model()
        history = temp_model.fit(
            train_inputs,
            train_targets,
            epochs=num_epochs,
            batch_size=512,
            validation_data=(validation_inputs, validation_targets),
            verbose=0,
        )
        history_dict = history.history
        Display.prototype_progress(history_dict)

    def k_fold_validation(self, k: int = 4, *, num_epochs: int = 50):
        indeces_permutation = np.random.permutation(len(self._array))
        shuffled_inputs = self._inputs[indeces_permutation]
        shuffled_targets = self._outputs[indeces_permutation]
        temp_model = Learning.pipeline_model()
        num_val_samples = len(shuffled_inputs) // k
        all_accuracy_scores = []
        for i in range(k):
            print(f"Processing fold #{i}")
            partial_train_data = np.concatenate(
                [
                    shuffled_inputs[: i * num_val_samples],
                    shuffled_inputs[(i + 1) * num_val_samples :],
                ],
                axis=0,
            )
            partial_train_targets = np.concatenate(
                [
                    shuffled_targets[: i * num_val_samples],
                    shuffled_targets[(i + 1) * num_val_samples :],
                ],
                axis=0,
            )
            val_data = shuffled_inputs[i * num_val_samples : (i + 1) * num_val_samples]
            val_targets = shuffled_targets[
                i * num_val_samples : (i + 1) * num_val_samples
            ]

            history = temp_model.fit(
                partial_train_data,
                partial_train_targets,
                epochs=num_epochs,
                batch_size=512,
                validation_data=(val_data, val_targets),
                verbose=0,
            )
            history_dict = history.history
            accuracy_history = history_dict["accuracy"]
            all_accuracy_scores.append(accuracy_history)

        score_mean = [
            np.mean([x[i] for x in all_accuracy_scores]) for i in range(num_epochs)
        ]
        Display.k_fold("Accuracy", score_mean)

    def finalize_model(self, *, num_epochs: int):
        indeces_permutation = np.random.permutation(len(self._array))
        shuffled_inputs = self._inputs[indeces_permutation]
        shuffled_targets = self._outputs[indeces_permutation]

        num_validation_samples = int(0.3 * len(self._inputs))
        train_inputs = shuffled_inputs[:num_validation_samples]
        train_targets = shuffled_targets[:num_validation_samples]
        validation_inputs = shuffled_inputs[num_validation_samples:]
        validation_targets = shuffled_targets[num_validation_samples:]

        self.model = Learning.pipeline_model()

        self.model.fit(
            train_inputs, train_targets, epochs=num_epochs, batch_size=512, verbose=0
        )
        self._performance_metrics = self.model.evaluate(
            validation_inputs, validation_targets, batch_size=512
        )


class Display:
    @staticmethod
    def prototype_progress(history_dict: dict):
        loss_values = history_dict["loss"]
        epochs = range(1, len(loss_values) + 1)
        val_loss_values = history_dict["val_loss"]
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.supxlabel("Epochs")
        fig.suptitle("Model Performance")
        ax1.plot(epochs, loss_values, "bo", label="Training Loss")
        ax1.plot(epochs, val_loss_values, "b", label="Validation Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2.plot(epochs, acc, "bo", label="Training Accuracy")
        ax2.plot(epochs, val_acc, "b", label="Validation Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        plt.show()

    @staticmethod
    def k_fold(metric: str, metric_scores: list[float]):
        plt.plot(range(1, len(metric_scores) + 1), metric_scores)
        plt.xlabel("Epochs")
        plt.ylabel(f"Average {metric}")
        plt.title(f"K-Fold Validation : {metric}")
        plt.show()


class Export:
    def __init__(self):
        file_name = input("Name model output file : ")
        self._save_file = os.path.join(os.getcwd(), file_name)
        self()

    def __call__(self, model):
        model.save(self._save_file)


if __name__ == "__main__":
    pre_processed_info = File().read_file().identify_parameters()
    digested_data = Digestion(pre_processed_info.data)
    digested_data()
    ml_modeling = Learning(
        digested_data.array,
        digested_data.shuffled_inputs,
        digested_data.shuffled_outputs,
    )
    ml_modeling.prototype_model(num_epochs=200)
    new_epochs = int(input("Choose the number of epochs : "))
    ml_modeling.k_fold_validation(num_epochs=new_epochs)
