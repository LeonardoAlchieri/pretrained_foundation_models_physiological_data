import numpy as np

class EDADataset:
    """
    A class representing the USI Laughs dataset.
    """

    def __init__(self, path_to_data: str, validation_method: callable, feature_extractor: callable):
        """
        Initialize the USI Laughs dataset.

        :param path_to_data: Path to the dataset.
        """
        self.path_to_data = path_to_data
        self.data = self._load_data(path_to_data)
        self.validation_method = validation_method
        self.extracted_features: bool = False
        self.feature_extractor = feature_extractor

        

    def _load_data(self, path: str) -> dict[str, np.ndarray]:
        """
        Load the USI Laughs dataset.
        This method should be implemented to load the actual dataset.
        """
        return np.load(path, allow_pickle=True)
    
    def extract_features(self, inplace: bool = False):
        """
        Extract features from the dataset using the provided feature extractor.
        """
        if not self.extracted_features:
            self.data = self.feature_extractor(self.data)
            self.extracted_features = True
        if not inplace:
            return self
    

    def train_test_split(self, inplace: bool = False):
        if not self.extracted_features:
            raise RuntimeError("Features must be extracted before splitting the dataset.")
        
        self.train_data, self.test_data = self.validation_method(self.data)
        if not inplace:
            return self