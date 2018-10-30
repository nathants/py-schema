import pytest
import util.dicts
import tornado.concurrent
import tornado.ioloop
import tornado.gen
from schema import validate, check

# TODO queues

# python specific tests

def test_bytes_matches_str_schemas():
    schema = 'asdf'
    validate(schema, b'asdf')

def test_bytes_not_synonymous_with_str():
    assert validate(bytes, b'123') == b'123'
    with pytest.raises(AssertionError):
        validate(str, b'123')

def test_unicode_synonymous_with_str():
    assert validate(str, u'asdf') == 'asdf'
    assert validate(u'asdf', 'asdf') == 'asdf'
    assert validate('asdf', u'asdf') == 'asdf'
    assert validate(dict, {u'a': 'b'}) == {'a': 'b'}

def test_fn_types():
    schema = (':fn', (int, int), {'returns': str})

    @check
    def fn(x: int, y: int) -> str:
        pass
    assert validate(schema, fn) is fn

    @check
    def fn(x: int, y: float) -> str:
        pass
    with pytest.raises(AssertionError):
        validate(schema, fn) # pos arg 2 invalid

    @check
    def fn(x: int, y: int) -> float:
        pass
    with pytest.raises(AssertionError):
        validate(schema, fn) # return invalid

    @check
    def fn(x: int, y: int):
        pass
    with pytest.raises(AssertionError):
        validate(schema, fn) # missing return schema

def test_annotations_return():
    def fn() -> str:
        return 123
    fn = check()(fn)
    with pytest.raises(AssertionError):
        fn()

def test_annotation_args():
    def fn(x: int) -> str:
        return str(x)
    fn = check()(fn)
    assert fn(1) == '1'
    with pytest.raises(AssertionError):
        fn(1.0)

def test_annotation_kwargs():
    def fn(x: int = 0) -> str:
        return str(x)
    fn = check()(fn)
    assert fn(x=1) == '1'
    with pytest.raises(AssertionError):
        fn(x=1.0)

def test_check_args_and_kwargs():
    @check
    def fn(a: int, b: float = 0) -> str:
        return str(a + b)
    assert fn(1) == '1'
    assert fn(1, b=.5) == '1.5'
    with pytest.raises(AssertionError):
        fn(1, 1)
    with pytest.raises(AssertionError):
        fn(1.0)
    with pytest.raises(AssertionError):
        fn(1, b='2')
    with pytest.raises(AssertionError):
        fn(1, c='2')

def test_check_returns():
    @check
    def badfn() -> str:
        return 0
    with pytest.raises(AssertionError):
        badfn()

def test_check_generators():
    @check
    def main(x: int):
        yield
    next(main(1))
    with pytest.raises(AssertionError):
        next(main(1.0))

def test_check_coroutines():
    @tornado.gen.coroutine
    @check
    def main(x: int) -> float:
        yield None
        if x > 0:
            x = float(x)
        return x
    assert tornado.ioloop.IOLoop.instance().run_sync(lambda: main(1)) == 1.0
    with pytest.raises(AssertionError):
        tornado.ioloop.IOLoop.instance().run_sync(lambda: main(1.0))
    with pytest.raises(AssertionError):
        tornado.ioloop.IOLoop.instance().run_sync(lambda: main(-1))

def test_check_yields_and_sends():
    @check(sends=int, yields=str)
    def main():
        val = yield 'a'
        if val > 0:
            yield 'b'
        else:
            yield 3

    gen = main()
    assert gen.send(None) == 'a'
    assert gen.send(1) == 'b'

    gen = main()
    next(gen)
    with pytest.raises(AssertionError):
        gen.send(-1) # violate yields

    gen = main()
    next(gen)
    with pytest.raises(AssertionError):
        gen.send('1') # violate _send

def test_method():
    class Foo(object):
        @check
        def bar(self, x: int) -> str:
            if x == 0:
                return 0
            else:
                return str(x)
    assert Foo().bar(1) == '1'
    with pytest.raises(AssertionError):
        Foo().bar('1')
    with pytest.raises(AssertionError):
        Foo().bar(0)

def test_generator_method():
    class Foo(object):
        @check(yields=str)
        def bar(self, x: int):
            if x == 0:
                yield 0
            else:
                yield str(x)
    assert next(Foo().bar(1)) == '1'
    with pytest.raises(AssertionError):
        next(Foo().bar('1'))
    with pytest.raises(AssertionError):
        next(Foo().bar(0))

def test_kwargs():
    @check
    def fn(**kw: {str: int}):
        assert 'a' in kw and 'b' in kw
        return True
    fn(a=1, b=2)
    with pytest.raises(AssertionError):
        fn(a=1, b=2.0)

def test_args():
    @check
    def fn(*a: [int]):
        return True
    fn(1, 2)
    with pytest.raises(AssertionError):
        fn(1, 2.0)

# common tests between python and clojure

def test_set_schema():
    schema = {int}
    assert validate(schema, {1, 2}) == {1, 2}

def test_none_as_schema():
    schema = {str: None}
    assert validate(schema, {'a': None}) == {'a': None}

def test_dict_behavior_key_ordering():
    schema = {int: float}
    assert validate(schema, {1: 1.1}) == {1: 1.1}
    assert validate(schema, {1.1: 1, 1: 1.1}) == {1: 1.1}
    assert validate(schema, {1: 1.1, 1.1: 1}) == {1: 1.1}

def test_false_as_schema():
    schema = {str: False}
    assert validate(schema, {'a': False}) == {'a': False}

def test_new_schema_old_data():
    schema = {'a': int, 'b': (':O', int, 2)}
    assert validate(schema, {'a': 1}) == {'a': 1, 'b': 2}

def test_old_schema_new_data():
    schema = {'a': int}
    assert validate(schema, {'a': 1, 'b': 2}) == {'a': 1}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 1, 'b': 2}, exact_match=True)

def test_exact_match():
    schema = {'a': 1}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 1, 'b': 2}, exact_match=True)

def test_missing_keys_in_value_are_never_allowed():
    schema = {'a': int, 'b': int}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 1}, exact_match=True)
    with pytest.raises(AssertionError):
        validate(schema, {'a': 1})

def test_type_schemas_pass_value_through():
    schema = object
    x = object()
    assert validate(schema, x) is x

def test_future():
    schema = str
    f = tornado.concurrent.Future()
    f = validate(schema, f)
    f.set_result('asdf')
    assert f.result() == 'asdf'

def test_future_fail():
    schema = str
    f = tornado.concurrent.Future()
    f = validate(schema, f)
    with pytest.raises(AssertionError):
        f.set_result(1)

def test_union():
    schema = (':U', str, None)
    assert validate(schema, 'foo') == 'foo'
    assert validate(schema, None) is None
    with pytest.raises(AssertionError):
        validate(schema, True)

def test_union_empty():
    schema = (':U',)
    with pytest.raises(AssertionError):
        validate(schema, True)

def test_intersection():
    schema = (':I', str, lambda x: len(x) > 2)
    assert validate(schema, 'foo') == 'foo'
    with pytest.raises(AssertionError):
        assert validate(schema, [])
    with pytest.raises(AssertionError):
        assert validate(schema, "")

def test_intersection_empty():
    schema = (':I',)
    with pytest.raises(AssertionError):
        validate(schema, True)

def test_union_applied_in_order():
    schema = (':U', {'name': (':O', str, 'bob')},
                    {'name': int})
    assert validate(schema, {}) == {'name': 'bob'}
    assert validate(schema, {'name': 123}) == {'name': 123}
    with pytest.raises(AssertionError):
        validate(schema, {'name': 1.0})

def test_intersection_applied_in_order():
    schema = (':I', {'name': (':O', object, 'bob')},
                    {'name': int})
    assert validate(schema, {'name': 123}) == {'name': 123}
    with pytest.raises(AssertionError):
        validate(schema, {})
    with pytest.raises(AssertionError):
        validate(schema, {'name': 'not-an-int'})

def test_predicate():
    schema = {str: callable}
    val = {'fn': lambda: None}
    assert validate(schema, val)['fn'] is val['fn']
    with pytest.raises(AssertionError):
        validate(schema, {'not-fn': None})

def test_predicate_keys_are_optional():
    schema = {lambda x: isinstance(x, str): str}
    assert validate(schema, {'a': 'b'}) == {'a': 'b'}
    assert validate(schema, {('not', 'a', 'str'): 'value-to-drop'}) == {}

def test_empty_dicts():
    assert validate({}, {}) == {}

def test_type_keys_are_optional():
    assert validate({str: str}, {}) == {}

def test_empty_dicts_exact_match():
    with pytest.raises(AssertionError):
        assert validate({}, {'1': 2}, True)

def test_partial_comparisons_for_testing():
    schema = {'blah': str,
              'data': [{str: str}]}
    data = {'blah': 'foobar',
            'data': [{'a': 'b'},
                     {'c': 'd'},
                     # ...
                     # pretend 'data' is something too large to specify as a value literal in a test
                     ]}
    validate(schema, data)
    with pytest.raises(AssertionError):
        validate(schema, {'blah': 'foobar',
                          'data': [{'a': 1}]})

def test_object_as_key():
    schema = {object: int}
    assert validate(schema, {'1': 2}) == {'1': 2}
    with pytest.raises(AssertionError):
        validate(schema, {'1': 2.0})

def test_object_tuple():
    schema = (object, object)
    assert validate(schema, (1, '2')) == [1, '2']
    with pytest.raises(AssertionError):
        validate(schema, (1, 2, 3))

def test_object_list():
    schema = [object]
    assert validate(schema, [1, 2, 3]) == [1, 2, 3]
    assert validate(schema, [1, '2', 3.0]) == [1, '2', 3.0]

def test_object_value():
    schema = {str: object}
    assert validate(schema, {'a': 'apple'}) == {'a': 'apple'}
    assert validate(schema, {'b': 123}) == {'b': 123}

def test_object_value_exact_match():
    schema = {str: object}
    assert validate(schema, {1: 'apple'}) == {}
    with pytest.raises(AssertionError):
        validate(schema, {1: 'apple'}, True)

def test_required_value_to_type():
    schema = {'a': 'apple',
              'b': str}
    assert validate(schema, {'a': 'apple', 'b': 'banana'}) == {'a': 'apple', 'b': 'banana'}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 'apple'})
    with pytest.raises(AssertionError):
        validate(schema, {'a': 'apple', 'b': 1})

def test_required_value_to_value():
    schema = {'a': 'apple',
              'b': 'banana'}
    assert validate(schema, {'a': 'apple', 'b': 'banana'}) == {'a': 'apple', 'b': 'banana'}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 'apple'})

def test_type_to_value():
    schema = {str: 'apple'}
    assert validate(schema, {'a': 'apple'}) == {'a': 'apple'}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 'notapple'})

def test_nested_optional():
    schema = {'a': {'b': (':O', object, 'default-val')}}
    assert validate(schema, {'a': {}}) == {'a': {'b': 'default-val'}}
    schema = [{'name': (':O', object, 'bob')}]
    assert validate(schema, [{}]) == [{'name': 'bob'}]

def test_optional():
    schema = {'a': 'apple',
              'b': [':O', str, 'banana']}
    assert validate(schema, {'a': 'apple'}) == {'a': 'apple', 'b': 'banana'}
    assert validate(schema, {'a': 'apple', 'b': 'bar'}) == {'a': 'apple', 'b': 'bar'}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 'apple', 'b': 1.0})

def test_value_schema():
    schema = 1
    assert validate(schema, 1) == 1
    with pytest.raises(AssertionError):
        validate(schema, 2)

def test_single_type_schema():
    schema = int
    assert validate(schema, 1) == 1
    with pytest.raises(AssertionError):
        validate(schema, '1')

def test_iterable_length_n():
    schema = [int]
    assert validate(schema, [1, 2]) == [1, 2]
    with pytest.raises(AssertionError):
        validate(schema, [1, '2'])

def test_iterable_fixed_length():
    schema = (float, int)
    assert validate(schema, [1.1, 2]) == [1.1, 2]
    with pytest.raises(AssertionError):
        validate(schema, [1.1, '2'])

def test_nested_type_to_type():
    schema = {str: {str: int}}
    assert validate(schema, {'1': {'1': 1}}) == {'1': {'1': 1}}
    with pytest.raises(AssertionError):
        validate(schema, {'1': None})
    with pytest.raises(AssertionError):
        validate(schema, {'1': {'1': None}})

def test_val_to_val_and_type_to_type():
    schema = {'a': 'apple',
              str: float}
    assert validate(schema, {'a': 'apple', '1': 1.1}) == {'a': 'apple', '1': 1.1}
    assert validate(schema, {'a': 'apple'}) == {'a': 'apple'}
    with pytest.raises(AssertionError):
        validate(schema, {'a': 'applebees'})

def test_type_to_type():
    schema = {str: int}
    assert validate(schema, {'1': 1}) == {'1': 1}
    assert validate(schema, {}) == {}
    with pytest.raises(AssertionError):
        validate(schema, {'1': '1'})

def test_value_to_type():
    schema = {'foo': int}
    assert validate(schema, {'foo': 1}) == {'foo': 1}
    with pytest.raises(AssertionError):
        validate(schema, {'foo': 'bar'})

def test_value_to_value():
    schema = {'foo': 'bar'}
    validate(schema, {'foo': 'bar'})
    with pytest.raises(AssertionError):
        validate(schema, {'foo': 1})

def test_predicate_schema():
    schema = {'foo': lambda x: isinstance(x, int) and x > 0}
    assert validate(schema, {'foo': 1}) == {'foo': 1}
    with pytest.raises(AssertionError):
        validate(schema, {'foo': 0})

def test_nested_predicate_schema():
    schema = {'foo': {'bar': lambda x: isinstance(x, int) and x > 0}}
    assert validate(schema, {'foo': {'bar': 1}}) == {'foo': {'bar': 1}}
    with pytest.raises(AssertionError):
        validate(schema, {'foo': {'bar': 0}})

def test_iterable_length_n_must_be_length_one():
    schema = [str, str]
    with pytest.raises(AssertionError):
        validate(schema, ['blah'])

def test_nested_iterables():
    schema = [[str]]
    assert validate(schema, [['1'], ['2']]) == [['1'], ['2']]
    with pytest.raises(AssertionError):
        assert validate(schema, [['1'], [1]])

def test_many_keys():
    schema = {str: int}
    assert validate(schema, {'1': 2, '3': 4}) == {'1': 2, '3': 4}
    with pytest.raises(AssertionError):
        validate(schema, {'1': 2, '3': 4.0})

def test_value_matches_are_higher_precedence_than_type_matches():
    schema = {str: int,
              'foo': 'bar'}
    assert validate(schema, {'1': 2, 'foo': 'bar'}) == {'1': 2, 'foo': 'bar'}
    with pytest.raises(AssertionError):
        validate(schema, {'1': 2, 'foo': 3})

def test_complex_types():
    schema = {'name': (str, str),
              'age': lambda x: isinstance(x, int) and x > 0,
              'friends': [(str, str)],
              'events': [{'what': str,
                          'when': float,
                          'where': (int, int)}]}
    data = {'name': ['jane', 'doe'],
            'age': 99,
            'friends': [['dave', 'g'],
                        ['tom', 'p']],
            'events': [{'what': 'party',
                        'when': 123.11,
                        'where': [65, 73]},
                       {'what': 'shopping',
                        'when': 145.22,
                        'where': [77, 44]}]}
    assert validate(schema, data) == data
    with pytest.raises(AssertionError):
        validate(schema, util.dicts.merge(data, {'name': 123}))
    with pytest.raises(AssertionError):
        validate(schema, util.dicts.merge(data, {'events': [None]}))
    with pytest.raises(AssertionError):
        validate(schema, util.dicts.merge(data, {'events': [None] + data['events']}))
    with pytest.raises(AssertionError):
        validate(schema, util.dicts.merge(data, {'events': [{'what': 'shopping',
                                                                     'when': 123.11,
                                                                     'where': [0]}]}))
