import logging

from swi_ml import logger as _global_logger
from swi_ml.backend import _Backend
from swi_ml import distributions

logger = logging.getLogger(__name__)


class NaiveBayesClassification(_Backend):
    def __init__(self, distribution="gaussian", verbose=None):
        if verbose is not None:
            logger.setLevel(verbose)
        else:
            logger.setLevel(_global_logger.getEffectiveLevel())

        if distribution.lower() == "gaussian":
            self.distribution = distributions.GaussianDistribution
        else:
            raise NotImplementedError(
                "Only 'gaussian' distribution is implemented yet"
            )
        self.unique_classes = None
        self._class_distributions = dict()
        self.backend = super().get_backend()

    def _calculate_class_prior(self, unique_class):
        """
        prior of unique_class
        (samples_with_class == unique_class / total_samples)
        """
        return self.backend.mean(self.unique_classes == unique_class)

    def _fit_preprocess(self, data, labels):
        # cast to array, CuPy backend will load the arrays on GPU
        X = self.backend.asarray(data)
        Y = self.backend.asarray(labels)
        return X, Y

    def fit(self, data, labels):
        """
        Learns the distribution of each label for each feature
        """
        X, Y = self._fit_preprocess(data, labels)
        self.unique_classes = self.backend.unique(Y).astype(int)
        logger.info(f"Unique Classes: {self.unique_classes.shape}")

        for label in self.unique_classes:
            data_subset = X[self.backend.where(Y == label)]
            # to prevent unhashable type (see https://github.com/cupy/cupy/issues/4531)
            label = int(label)
            # bootstrap class distribution as list (of distributions)
            self._class_distributions[label] = []
            for feature in data_subset.T:
                # get distribution from mean and variance
                dist = self.distribution(feature.mean(), feature.var())
                logger.info(f"Label: {label}, Distribution: {dist}")
                self._class_distributions[label].append(dist)

    def _predict_preprocess(self, data, probability):
        data = self.backend.asarray(data)
        if len(data.shape) < 2:
            data = data.reshape(1, -1)
        if probability:
            predictions = self.backend.empty((data.shape[0], 2))
        else:
            predictions = self.backend.empty(data.shape[0])
        return data, predictions

    def predict(self, data, probability=True):
        """
        Classifies samples according to the largest probability
        with the learnt distributions
        """
        data, predictions = self._predict_preprocess(data, probability)
        for idx, sample in enumerate(data):
            class_probabilities = self.backend.empty(self.unique_classes.shape)
            for unique_class in self.unique_classes:
                posteriors = self._calculate_class_prior(unique_class)
                # to prevent unhashable type (see https://github.com/cupy/cupy/issues/4531)
                unique_class = int(unique_class)
                class_distributions = self._class_distributions[unique_class]
                for sample_value, dist in zip(sample, class_distributions):
                    posteriors *= dist.pdf(sample_value)
                class_probabilities[unique_class] = posteriors
            if probability:
                predictions[idx] = class_probabilities
            else:
                predictions[idx] = self.unique_classes[
                    self.backend.argmax(class_probabilities)
                ]
        return predictions
