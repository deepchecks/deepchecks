from hamcrest import assert_that, equal_to, is_not

from deepchecks.nlp.utils.text_properties import TOXICITY_MODEL_NAME
from deepchecks.nlp.utils.text_properties_models import get_transformer_pipeline


def test_model_caching():
    model1 = get_transformer_pipeline(
        property_name='toxicity', model_name=TOXICITY_MODEL_NAME, device=None,
        models_storage=None, use_cache=True)

    model2 = get_transformer_pipeline(
        property_name='toxicity', model_name=TOXICITY_MODEL_NAME, device=None,
        models_storage=None, use_cache=True)

    model3 = get_transformer_pipeline(
        property_name='toxicity', model_name=TOXICITY_MODEL_NAME, device=None,
        models_storage=None, use_cache=False)

    assert_that(id(model1), equal_to(id(model2)))
    assert_that(id(model1), is_not(equal_to(id(model3))))
    assert_that(id(model2), is_not(equal_to(id(model3))))
