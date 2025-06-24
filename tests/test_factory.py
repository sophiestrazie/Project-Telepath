# tests/test_factory.py
def test_classifier_creation():
    from models import ClassifierFactory
    svm = ClassifierFactory.create_classifier('svm', {'C': 1.0})
    assert hasattr(svm, 'fit') and hasattr(svm, 'predict')