from exceptions import Exception


def exception_factory(exception_name, base_exception=Exception, attributes=None):
    attribute_dict = {
        "__init__": base_exception.__init__
    }
    if isinstance(attributes, dict):
        attributes.update(attributes)
    return type(
        exception_name,
        (base_exception, ),
        attribute_dict
    )


DataSetNotLoadedError = exception_factory('DataSetNotLoadedError')
InputXError = exception_factory('InputXError', base_exception=ValueError)
InputYError = exception_factory('InputYError', base_exception=ValueError)
InputXYError = exception_factory('InputXYError', base_exception=ValueError)
InputGroupbyDfError = exception_factory('InputGroupbyDfError', base_exception=ValueError)
LabelShapeError = exception_factory('LabelShapeError', base_exception=ValueError)
FeatureNamesShapeError = exception_factory('FeatureNamesShapeError', base_exception=LabelShapeError)