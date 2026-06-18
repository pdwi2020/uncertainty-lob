try:
    from .hmm_detector import HMMRegimeDetector
except ImportError:
    HMMRegimeDetector = None  # type: ignore  # hmmlearn not installed locally; available on Kaggle
