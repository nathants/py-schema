import pytest
import s.dicts
import schema
import tornado.concurrent
import tornado.ioloop
import tornado.gen


def test_none_as_schema():
    shape = {str: None}
    schema.validate(shape, {'a': None})


def test_none_as_schema():
    shape = {str: False}
    schema.validate(shape, {'a': False})


def test_new_schema_old_data():
    shape = {'a': int, 'b': (':O', int, 2)}
    assert schema.validate(shape, {'a': 1}) == {'a': 1, 'b': 2}


def test_old_schema_new_data():
    shape = {'a': int}
    assert schema.validate(shape, {'a': 1, 'b': 2}) == {'a': 1}


def test_exact_match():
    shape = {'a': 1}
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 1, 'b': 2}, True)


def test_missing_keys_in_value_are_never_allowed():
    shape = {'a': int, 'b': int}
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 1}, True)
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 1})


def test_future_when_the_schema_is_a_future_type():
    shape = tornado.concurrent.Future
    f = shape()
    assert schema.validate(shape, f) is f


def test_future():
    shape = str
    f1 = tornado.concurrent.Future()
    f2 = schema.validate(shape, f1)
    assert f1 is not f2
    f1.set_result('asdf')
    assert f2.result() == 'asdf'


def test_future_fail():
    shape = str
    f1 = tornado.concurrent.Future()
    f2 = schema.validate(shape, f1)
    assert f1 is not f2
    f1.set_result(1)
    with pytest.raises(schema.Error):
        f2.result()


def test_maybe():
    shape = (':M', str)
    assert schema.validate(shape, 'foo') == 'foo'
    assert schema.validate(shape, None) is None
    with pytest.raises(schema.Error):
        schema.validate(shape, True)


def test_method():
    class Foo(object):
        @schema.check
        def bar(self, x: int) -> str:
            if x == 0:
                return 0
            else:
                return str(x)
    assert Foo().bar(1) == '1'
    with pytest.raises(Exception):
        Foo().bar('1')
    with pytest.raises(Exception):
        Foo().bar(0)


def test_generator_method():
    class Foo(object):
        @schema.check(yields=str)
        def bar(self, x: int):
            if x == 0:
                yield 0
            else:
                yield str(x)
    assert next(Foo().bar(1)) == '1'
    with pytest.raises(Exception):
        next(Foo().bar('1'))
    with pytest.raises(Exception):
        next(Foo().bar(0))


def test_kwargs():
    @schema.check
    def fn(**kw: {str: int}):
        assert 'a' in kw and 'b' in kw
        return True
    fn(a=1, b=2)
    with pytest.raises(schema.Error):
        fn(a=1, b=2.0)


def test_args():
    @schema.check
    def fn(*a: [int]):
        return True
    fn(1, 2)
    with pytest.raises(schema.Error):
        fn(1, 2.0)


def test_callable():
    shape = {str: callable}
    val = {'fn': lambda: None}
    assert schema.validate(shape, val)['fn'] is val['fn']
    with pytest.raises(Exception):
        schema.validate(shape, {'not-fn': None})


def test_fn_types():
    shape = (':fn', (int, int), {'returns': str})

    @schema.check
    def fn(x: int, y: int) -> str:
        pass
    assert schema.validate(shape, fn) is fn

    @schema.check
    def fn(x: int, y: float) -> str:
        pass
    with pytest.raises(schema.Error):
        schema.validate(shape, fn) # pos arg 2 invalid

    @schema.check
    def fn(x: int, y: int) -> float:
        pass
    with pytest.raises(schema.Error):
        schema.validate(shape, fn) # return invalid

    @schema.check
    def fn(x: int, y: int):
        pass
    with pytest.raises(schema.Error):
        schema.validate(shape, fn) # missing return shape


def test_union_types():
    shape = (':U', int, float)
    assert schema.validate(shape, 1) == 1
    assert schema.validate(shape, 1.0) == 1.0
    with pytest.raises(schema.Error):
        schema.validate((':U', int, float), '1')

    shape = (':U', [int], {str: int})
    assert schema.validate(shape, [1]) == [1]
    assert schema.validate(shape, {'1': 2}) == {'1': 2}
    with pytest.raises(schema.Error):
        schema.validate(shape, [1.0])
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': 2.0})


def test_sets_are_illegal():
    with pytest.raises(schema.Error):
        schema.validate({1, 2}, set())


def test_empty_dicts():
    assert schema.validate({}, {}) == {}
    assert schema.validate({str: str}, {}) == {}


def test_empty_dicts_exact_match():
    with pytest.raises(schema.Error):
        assert schema.validate({}, {'1': 2}, True)


def test_empty_seqs():
    assert schema.validate(list, []) == []
    assert schema.validate(tuple, ()) == ()
    assert schema.validate([str], []) == []
    with pytest.raises(schema.Error):
        schema.validate([], [123])
    with pytest.raises(schema.Error):
        schema.validate([], (123,))


def test_validate_returns_value():
    assert schema.validate(int, 123) == 123


def test_unicde_synonymous_with_str():
    assert schema.validate(str, u'asdf') == 'asdf'
    assert schema.validate(u'asdf', 'asdf') == 'asdf'
    assert schema.validate('asdf', u'asdf') == 'asdf'
    assert schema.validate(dict, {u'a': 'b'}) == {'a': 'b'}


def test_bytes_not_synonymous_with_str():
    assert schema.validate(bytes, b'123') == b'123'
    with pytest.raises(schema.Error):
        schema.validate(str, b'123')


def test_bytes_matches_str_schemas():
    shape = 'asdf'
    schema.validate(shape, b'asdf')


def test_partial_comparisons_for_testing():
    shape = {'blah': str,
             'data': [{str: str}]}
    data = {'blah': 'foobar',
            'data': [{'a': 'b'},
                     {'c': 'd'},
                     # ...
                     # pretend 'data' is something too large to specify as a value literal in a test
                     ]}
    schema.validate(shape, data)
    with pytest.raises(schema.Error):
        schema.validate(shape, {'blah': 'foobar',
                                'data': [{'a': 1}]})


def test_object_dict():
    shape = {object: int}
    schema.validate(shape, {'1': 2})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': 2.0})


def test_object_tuple():
    shape = (object, object)
    schema.validate(shape, (1, '2'))
    with pytest.raises(schema.Error):
        schema.validate(shape, (1, 2, 3))


def test_object_list():
    shape = [object]
    schema.validate(shape, [1, 2, 3])
    schema.validate(shape, [1, '2', 3.0])


def test_annotations_return():
    def fn() -> str:
        return 123
    fn = schema.check()(fn)
    with pytest.raises(schema.Error):
        fn()


def test_annotation_args():
    def fn(x: int) -> str:
        return str(x)
    fn = schema.check()(fn)
    assert fn(1) == '1'
    with pytest.raises(schema.Error):
        fn(1.0)


def test_annotation_kwargs():
    def fn(x: int = 0) -> str:
        return str(x)
    fn = schema.check()(fn)
    assert fn(x=1) == '1'
    with pytest.raises(schema.Error):
        fn(x=1.0)


def test_check_args_and_kwargs():
    @schema.check
    def fn(a: int, b: float = 0) -> str:
        return str(a + b)
    assert fn(1) == '1'
    assert fn(1, b=.5) == '1.5'
    with pytest.raises(schema.Error):
        fn(1, 1)
    with pytest.raises(schema.Error):
        fn(1.0)
    with pytest.raises(schema.Error):
        fn(1, b='2')
    with pytest.raises(schema.Error):
        fn(1, c='2')


def test_check_returns():
    @schema.check
    def badfn() -> str:
        return 0
    with pytest.raises(schema.Error):
        badfn()


def test_check_generators():
    @schema.check
    def main(x: int):
        yield
    next(main(1))
    with pytest.raises(schema.Error):
        next(main(1.0))


def test_check_coroutines():
    @tornado.gen.coroutine
    @schema.check
    def main(x: int) -> float:
        yield tornado.gen.moment
        if x > 0:
            x = float(x)
        return x
    assert tornado.ioloop.IOLoop.instance().run_sync(lambda: main(1)) == 1.0
    with pytest.raises(schema.Error):
        tornado.ioloop.IOLoop.instance().run_sync(lambda: main(1.0))
    with pytest.raises(schema.Error):
        tornado.ioloop.IOLoop.instance().run_sync(lambda: main(-1))


def test_check_yields_and_sends():
    @schema.check(sends=int, yields=str)
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
    with pytest.raises(schema.Error):
        gen.send(-1) # violate yields

    gen = main()
    next(gen)
    with pytest.raises(schema.Error):
        gen.send('1') # violate _send


def test_object_type():
    shape = {str: object}
    schema.validate(shape, {'a': 'apple'})
    schema.validate(shape, {'b': 123})


def test_object_type_exact_match():
    shape = {str: object}
    with pytest.raises(schema.Error):
        schema.validate(shape, {1: 'apple'}, True)


def test_type_to_lambda():
    shape = {str: lambda x: x == 'apple'}
    schema.validate(shape, {'a': 'apple'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'notapple'})


def test_required_type_to_type():
    shape = {'a': 'apple',
             str: float}
    schema.validate(shape, {'a': 'apple', '1': 1.1})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple'})


def test_required_value_to_type():
    shape = {'a': 'apple',
             'b': str}
    schema.validate(shape, {'a': 'apple', 'b': 'banana'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple', 'b': 1})


def test_required_value_to_value():
    shape = {'a': 'apple',
             'b': 'banana'}
    schema.validate(shape, {'a': 'apple', 'b': 'banana'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple'})


def test_required_type_to_value():
    shape = {'a': 'apple',
             str: 'banana'}
    schema.validate(shape, {'a': 'apple', 'b': 'banana'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple', 1: 'banana'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple', 'b': 'notbanana'})


def test_type_to_value():
    shape = {str: 'apple'}
    schema.validate(shape, {'a': 'apple'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'notapple'})


def test_nested_optional():
    shape = {'a': {'b': (':O', object, 'default-val')}}
    assert schema.validate(shape, {'a': {}}) == {'a': {'b': 'default-val'}}
    shape = [{'name': (':O', object, 'bob')}]
    assert schema.validate(shape, [{}]) == [{'name': 'bob'}]


def test_optional_value_key_with_validation():
    shape = {'a': 'apple',
             'b': [':O', str, 'banana']}
    schema.validate(shape, {'a': 'apple'}) == {'a': 'apple', 'b': 'banana'}
    schema.validate(shape, {'a': 'apple', 'b': 'banana'}) == {'a': 'apple', 'b': 'banana'}
    with pytest.raises(schema.Error):
        schema.validate(shape, {'a': 'apple', 'b': 1.0})


def test_value_schema():
    shape = 1
    schema.validate(shape, 1)
    with pytest.raises(schema.Error):
        schema.validate(shape, 2)


def test_single_type_schema():
    shape = int
    schema.validate(shape, 1)
    with pytest.raises(schema.Error):
        schema.validate(shape, '1')


def test_single_iterable_length_n():
    shape = [int]
    schema.validate(shape, [1, 2])
    with pytest.raises(schema.Error):
        schema.validate(shape, [1, '2'])


def test_single_iterable_fixed_length():
    shape = (float, int)
    schema.validate(shape, [1.1, 2])
    with pytest.raises(schema.Error):
        schema.validate(shape, [1.1, '2'])


def test_nested_type_to_type_mismatch():
    shape = {str: {str: int}}
    schema.validate(shape, {'1': {'1': 1}})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': None})


def test_nested_type_to_type():
    shape = {str: {str: int}}
    schema.validate(shape, {'1': {'1': 1}})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': {'1': '1'}})


def test_type_to_type():
    shape = {str: int}
    schema.validate(shape, {'1': 1})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': '1'})


def test_value_to_type():
    shape = {'foo': int}
    schema.validate(shape, {'foo': 1})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'foo': 'bar'})


def test_value_to_value():
    shape = {'foo': 'bar'}
    schema.validate(shape, {'foo': 'bar'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'foo': 1})


def test_value_to_validator():
    shape = {'foo': lambda x: isinstance(x, int) and x > 0}
    schema.validate(shape, {'foo': 1})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'foo': 0})


def test_nested_value_to_validator():
    shape = {'foo': {'bar': lambda x: isinstance(x, int) and x > 0}}
    schema.validate(shape, {'foo': {'bar': 1}})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'foo': {'bar': 0}})


def test_iterable_length_n_bad_validator():
    shape = {str: [str, str]}
    with pytest.raises(schema.Error):
        schema.validate(shape, {'blah': ['blah', 'blah']})


def test_iterable_length_n():
    shape = {str: [str]}
    schema.validate(shape, {'1': ['1', '2']})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': 1})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': ['1', 2]})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': None})


def test_iterable_fixed_length():
    shape = {str: (str, str)}
    schema.validate(shape, {'1': ['1', '2']})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': ['1']})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': ['1', '2', '3']})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': ['1', 2]})


def test_nested_iterables():
    shape = {str: [[str]]}
    schema.validate(shape, {'1': [['1'], ['2']]})
    with pytest.raises(schema.Error):
        assert schema.validate(shape, {'1': [['1'], [1]]})


def test_many_keys():
    shape = {str: int}
    schema.validate(shape, {'1': 2, '3': 4})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': 2, '3': 4.0})


def test_value_matches_are_higher_precedence_than_type_matches():
    shape = {str: int,
             'foo': 'bar'}
    schema.validate(shape, {'1': 2, 'foo': 'bar'})
    with pytest.raises(schema.Error):
        schema.validate(shape, {'1': 2, 'foo': 'asdf'})


def test_complex_types():
    shape = {'name': (str, str),
             'age': lambda x: isinstance(x, int) and x > 0,
             'friends': [lambda x: isinstance(x, str) and len(x.split()) == 2],
             'events': [{'what': str,
                         'when': float,
                         'where': (int, int)}]}
    data = {'name': ('jane', 'doe'),
            'age': 99,
            'friends': ['dave g', 'tom p'],
            'events': [{'what': 'party',
                        'when': 123.11,
                        'where': (65, 73)},
                       {'what': 'shopping',
                        'when': 145.22,
                        'where': [77, 44]}]}
    schema.validate(shape, data)
    with pytest.raises(schema.Error):
        schema.validate(shape, s.dicts.merge(data, {'name': 123}))
    with pytest.raises(schema.Error):
        schema.validate(shape, s.dicts.merge(data, {'events': [None]}))
    with pytest.raises(schema.Error):
        schema.validate(shape, s.dicts.merge(data, {'events': [None] + data['events']}))
    with pytest.raises(schema.Error):
        schema.validate(shape, s.dicts.merge(data, {'events': [{'what': 'shopping',
                                                                'when': 123.11,
                                                                'where': [0]}]}))
