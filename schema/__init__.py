import functools
import inspect
import pprint
import re
import util.misc
import util.dicts
import util.exceptions
import util.iter
import util.func
import util.strings
import sys
import traceback
import types
import os

disabled = os.environ.get('SCHEMA_DISABLE')
if disabled:
    print('schema has been disabled', file=sys.stderr)

_schema_commands = (':or',
                    ':and',
                    ':optional',
                    ':fn')

def is_valid(schema, value):
    try:
        _validate(schema, value)
        return True
    except AssertionError:
        return False

def validate(schema, value, exact_match=False):
    """
    >>> import pytest

    ### basic usage

    # simple values represent themselves
    >>> schema = int
    >>> assert validate(schema, 123) == 123
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, '123')

    # lists represent variable length homogenous lists/tuples
    >>> schema = [int]
    >>> assert validate(schema, [1, 2]) == [1, 2]
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, [1, '2'])

    # tuples represent fixed length heterogenous lists/tuples
    >>> schema = (int, int)
    >>> assert validate(schema, [1, 2]) == [1, 2]
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, [1])

    ### union types with :or
    >>> schema = (':or', int, float)
    >>> assert validate(schema, 1) == 1
    >>> assert validate(schema, 1.0) == 1.0
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, '1')

    ## intersection types with :and
    >>> schema = (':and', lambda x: x.startswith('a'), lambda x: x.endswith('z'))
    >>> assert validate(schema, 'a-z') == 'a-z'
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, 'a-b')
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, 'b-z')

    ### dicts can use types and values for k's and v's, and also lambdas for v'util.

    # dicts with types->types
    >>> schema = {str: int}
    >>> assert validate(schema, {'1': 2}) == {'1': 2}
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, {'1': 2.0})

    # dicts with types->values. fyi, the only type allowed for keys is "str".
    >>> schema = {str: 'bob'}
    >>> assert validate(schema, {'alias': 'bob'}) == {'alias': 'bob'}
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, {'alias': 'joe'})

    # dicts with values->types
    >>> schema = {'name': float}
    >>> assert validate(schema, {'name': 3.14}) == {'name': 3.14}
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, {'name': 314})

    # dicts with complex validation
    >>> assert validate({'name': lambda x: x in ['john', 'jane']}, {'name': 'jane'}) == {'name': 'jane'}
    >>> with pytest.raises(AssertionError):
    ...     validate({'name': lambda x: x in ['john', 'jane']}, {'name': 'rose'})

    # dicts with :optional k's provide a value for a missing key and validate provided keys
    >>> schema = {'name': (':optional', str, 'jane')}
    >>> assert validate(schema, {}) == {'name': 'jane'}
    >>> assert validate(schema, {'name': 'rose'}) == {'name': 'rose'}
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, {'name': 123})

    # dicts with only type keys can be empty
    >>> schema = {str: str}
    >>> assert validate(schema, {}) == {}

    # validate is recursive, so nest schemas freely
    >>> schema = {'users': [{'name': (str, str), 'id': int}]}
    >>> obj = {'users': [{'name': ['jane', 'smith'], 'id': 85},
    ...                  {'name': ['john', 'smith'], 'id': 93}]}
    >>> assert validate(schema, obj) == obj
    >>> with pytest.raises(AssertionError):
    ...     validate(schema, {'users': [{'name': ('jane', 'e', 'smith'), 'id': 85}]})

    ### schema based pattern matching

    # # with a combination of values and object, we can express complex assertions on data
    # while True:
    #     msg = socket.recv()
    #     if validate([":order", {'sender': str, 'instructions': [str]], msg):
    #         key, val = msg
    #         run_order(val)
    #     elif validate([":shutdown", object]):
    #         sys.exit(1)
    #     else:
    #         print('unknown message')
    #
    """
    if disabled:
        return value
    return _validate(schema, value, exact_match)

def _validate(schema, value, exact_match=False):
    # maybe use ':type/<type>' instead of literal types? ie non jsonable stuff.
    with util.exceptions.update(_updater(schema, value), AssertionError):
        # TODO break this up into well named pieces
        # TODO replace long lists of conditionals with type based lookup in dicts. falls back on isinstance based looks? ugh. subclasses.
        value_is_a_future = util.misc.is_future(value)
        schema_is_a_future_type = util.misc.is_future(schema) and type(schema) is type
        if value_is_a_future and not schema_is_a_future_type:
            _set_result = value.set_result
            def f(x):
                _set_result(_validate(schema, x))
            value.set_result = f
            return value
        elif isinstance(schema, set):
            assert isinstance(value, set), '{} <{}> does not match schema: {} <{}>'.format(value, type(value), schema, type(schema))
            assert len(schema) == 1, 'set schemas represent homogenous sets and must contain a single schema: {}'.format(schema)
            return {_validate(list(schema)[0], x) for x in value}
        elif isinstance(schema, dict):
            assert isinstance(value, dict), '{} <{}> does not match schema: {} <{}>'.format(value, type(value), schema, type(schema))
            # if schema keys are all types, and _value is empty, return. ie, type keys are optional, so {} is a valid {int: int}
            if value == {} and {type(x) for x in schema} == {type}:
                return value
            else:
                # check for items in value that dont satisfy schema, dropping unknown keys unless exact_match=true
                # TODO update to conform to clj-schema. value, type, etc now deprecated.
                _value = value.copy()
                _value.clear()
                for k, v in value.items():
                    value_match = k in schema
                    type_match = type(k) in [x for x in schema if isinstance(x, type)] # TODO sort this comprehension for consistent results?
                    predicate_match = (list(filter(lambda f: f(k) and f, [x for x in schema if isinstance(x, (types.FunctionType, type(callable)))])) or [False])[0] # TODO sort this comprehension for consistent results?
                    any_match = object in schema
                    if value_match or type_match or predicate_match or any_match:
                        _schema = schema[k if value_match else
                                         type(k) if type_match else
                                         predicate_match if predicate_match else
                                         object]
                        _value[k] = _validate(_schema, v)
                    elif exact_match:
                        raise AssertionError('{} <{}> does not match schema keys: {}'.format(k, type(k), ', '.join(['{} <{}>'.format(x, type(x)) for x in schema])))
                # check for items in schema missing in value, filling in optional value
                for k, v in schema.items():
                    if k not in _value:
                        if isinstance(v, (list, tuple)) and v and v[0] == ':optional':
                            assert len(v) == 3, ':optional schema should be [:optional, schema, default-value], not: {}'.format(v)
                            _value[k] = _validate(*v[1:])
                        elif not (isinstance(k, type) or isinstance(k, (types.FunctionType, type(callable)))):
                            raise AssertionError('{} <{}> is missing required key: {} <{}>'.format(_value, type(_value), k, type(k)))
                return _value
        elif schema is object:
            return value
        # TODO flatten this into un-nested conditionals. perf hit?
        elif isinstance(schema, (list, tuple)):
            assert isinstance(value, (list, tuple)) or _starts_with_keyword(schema), '{} <{}> is not a seq: {} <{}>'.format(value, type(value), schema, type(schema))
            if schema and schema[0] in _schema_commands:
                if schema[0] == ':optional':
                    assert len(schema) == 3, ':optional schema should be [:optional, schema, default-value], not: {}'.format(schema)
                    return _validate(schema[1], value)
                elif schema[0] == ':or':
                    assert schema[1:], 'union types cannot be empty: {}'.format(schema)
                    tracebacks = []
                    for _schema in schema[1:]:
                        try:
                            value = _validate(_schema, value)
                        except AssertionError:
                            tracebacks.append(traceback.format_exc())
                    if len(tracebacks) == len(schema[1:]):
                        raise AssertionError('{} <{}> did not match *any* of [{}]\n{}'.format(value, type(value), ', '.join(['{} <{}>'.format(x, type(x)) for x in schema[1:]]), '\n'.join(tracebacks)))
                    else:
                        return value
                elif schema[0] == ':and':
                    assert schema[1:], 'intersection types cannot be empty: {}'.format(schema)
                    tracebacks = []
                    for _schema in schema[1:]:
                        try:
                            value = _validate(_schema, value)
                        except AssertionError:
                            tracebacks.append(traceback.format_exc())
                    if tracebacks:
                        raise AssertionError('{} <{}> did not match *all* of [{}]\n{}'.format(value, type(value), ', '.join(['{} <{}>'.format(x, type(x)) for x in schema[1:]]), '\n'.join(tracebacks)))
                    else:
                        return value
                elif schema[0] == ':fn':
                    assert isinstance(value, types.FunctionType), '{} <{}> is not a function'.format(value, type(value))
                    assert len(schema) in [2, 3], ':fn schema should be (:fn, [<args>...], {<kwargs>: <val>, ...}) or (:fn, [<args>...]), not: {}'.format(schema)
                    args, kwargs = schema[1:]
                    _args, _kwargs = value._schema # TODO have a sane error message here when ._schema undefined
                    assert tuple(_args) == tuple(args), 'pos args {_args} did not match {args}'.format(**locals())
                    assert _kwargs == kwargs, 'kwargs {_kwargs} did not match {kwargs}'.format(**locals())
                    return value
            elif isinstance(schema, list):
                assert len(schema) == 1, 'list schemas represent homogenous seqs and must contain a single schema: {}'.format(schema)
                return [_validate(schema[0], v) for v in value]
            elif isinstance(schema, tuple):
                assert len(schema) == len(value), '{} <{}> mismatched length of schema: {} <{}>'.format(value, type(value), schema, type(schema))
                return [_validate(_schema, _value) for _schema, _value in zip(schema, value)]
        elif isinstance(schema, type):
            assert isinstance(value, schema), '{} <{}> is not a: {} <{}>'.format(value, type(value), schema, type(schema))
            return value
        elif isinstance(schema, (types.FunctionType, type(callable))):
            assert schema(value), '{} <{}> failed predicate schema: {} <{}>'.format(value, type(value), util.func.source(schema), type(schema))
            return value
        else:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            assert value == schema, '{} <{}> does not equal: {} <{}>'.format(value, type(value), schema, type(schema))
            return value

def _formdent(x):
    return util.strings.indent(pprint.pformat(x, width=1), 2)

def _update_functions(schema):
    def fn(x):
        if isinstance(x, types.FunctionType):
            filename, linenum = x.__code__.co_filename, x.__code__.co_firstlineno
            x = 'lambda:{filename}:{linenum}'.format(**locals())
        return x
    return fn

def _updater(schema, value):
    return lambda x: _prettify(x + _helpful_message(schema, value))

def _helpful_message(schema, value):
    for fn in [x for x in util.iter.flatten(schema) if isinstance(x, (types.FunctionType, types.LambdaType))]:
        try:
            filename, linenum = fn.__code__.co_filename, fn.__code__.co_firstlineno
            with open(filename) as f:
                lines = f.read().splitlines()
            start = end = None
            for i in reversed(range(linenum)):
                if not lines[i].strip() or 'def ' in lines[i] or 'class ' in lines[i]:
                    break
                elif ' = ' in lines[i]:
                    start = i
                    break
            if start is None:
                filename, linenum = fn.__code__.co_filename, fn.__code__.co_firstlineno
                schema = 'function:{filename}:{linenum}'.format(**locals())
            else:
                if any(x in lines[start] for x in ['{', '(', '[']):
                    for i in range(linenum, len(lines) + 1):
                        text = '\n'.join(lines[start:i])
                        if all(text.count(x) == text.count(y) for x, y in [('{', '}'), ('[', ']'), ('(', ')')]):
                            end = i
                            break
                if end is not None:
                    schema = '\n'.join(lines[start:end])
                    size = len(lines[start]) - len(lines[start].lstrip())
                    schema = util.strings.unindent(schema, size)
            break
        except:
            continue
    else:
        schema = pprint.pformat(schema, width=1)
    return '\n\nobj:\n{}\nschema:\n{}'.format(
        util.strings.indent(pprint.pformat(value, width=1), 2),
        util.strings.indent(schema, 2),
    )

def _starts_with_keyword(x):
    if x and isinstance(x[0], str) and x[0].startswith(':'):
        return True
    else:
        return False

def _prettify(x):
    return re.sub(r"\<\w+ \'([\w\.]+)\'\>", r'\1', str(x))

def _get_schemas(fn, args, kwargs):
    arg_schemas, kwarg_schemas, return_schema = _read_annotations(fn, args, kwargs)
    schemas = {'yields': kwarg_schemas.pop('yields', object),
               'sends': kwarg_schemas.pop('sends', object),
               'returns': kwarg_schemas.pop('returns', return_schema),
               'args': kwarg_schemas.pop('args', None),
               'kwargs': kwarg_schemas.pop('kwargs', None),
               'arg': arg_schemas,
               'kwarg': kwarg_schemas}
    return schemas

def _read_annotations(fn, arg_schemas, kwarg_schemas):
    if not arg_schemas:
        sig = inspect.signature(fn)
        arg_schemas = [x.annotation
                       for x in sig.parameters.values()
                       if x.default is inspect._empty
                       and x.annotation is not inspect._empty
                       and x.kind is x.POSITIONAL_OR_KEYWORD]
    val = {x.name: x.annotation
           for x in sig.parameters.values()
           if x.default is not inspect._empty
           or x.kind is x.KEYWORD_ONLY
           and x.annotation is not inspect._empty}
    val = util.dicts.merge(val, {'args': x.annotation
                                 for x in sig.parameters.values()
                                 if x.annotation is not inspect._empty
                                 and x.kind is x.VAR_POSITIONAL})
    val = util.dicts.merge(val, {'kwargs': x.annotation
                                 for x in sig.parameters.values()
                                 if x.annotation is not inspect._empty
                                 and x.kind is x.VAR_KEYWORD})
    kwarg_schemas = util.dicts.merge(kwarg_schemas, val)
    if sig.return_annotation is not inspect._empty:
        return_schema = sig.return_annotation
    else:
        return_schema = object
    return arg_schemas, kwarg_schemas, return_schema

def _check_args(args, kwargs, name, schemas):
    with util.exceptions.update(_prettify, AssertionError):
        # TODO better to use inspect.getcallargs() for this? would change the semantics of pos arg checking. hmmn...
        # look at the todo in util.web.post for an example.
        assert len(schemas['arg']) == len(args) or schemas['args'], 'you asked to check {} for {} pos args, but {} were provided\nargs:\n{}\nschema:\n{}'.format(
            name, len(schemas['arg']), len(args), pprint.pformat(args, width=1), pprint.pformat(schemas, width=1)
        )
        _args = []
        for i, (schema, arg) in enumerate(zip(schemas['arg'], args)):
            with util.exceptions.update('pos arg num:\n  {}'.format(i), AssertionError):
                _args.append(validate(schema, arg))
        if schemas['args'] and args[len(schemas['arg']):]:
            _args += validate(schemas['args'], args[len(schemas['arg']):])
        _kwargs = {}
        for k, v in kwargs.items():
            if k in schemas['kwarg']:
                with util.exceptions.update('keyword arg:\n  {}'.format(k), AssertionError):
                    _kwargs[k] = validate(schemas['kwarg'][k], v)
            elif schemas['kwargs']:
                with util.exceptions.update('keyword args schema failed.', AssertionError):
                    _kwargs[k] = validate(schemas['kwargs'], {k: v})[k]
            else:
                raise AssertionError('cannot check {} for unknown key: {}={}'.format(name, k, v))
        return _args, _kwargs

def _fn_check(decoratee, name, schemas):
    @functools.wraps(decoratee)
    def decorated(*args, **kwargs):
        with util.exceptions.update('schema.check failed for args to function:\n  {}'.format(name), AssertionError, when=lambda x: 'failed for ' not in x):
            if args and decoratee.__code__ is getattr(getattr(args[0], decoratee.__name__, None), '__orig_code__', None):
                a, kwargs = _check_args(args[1:], kwargs, name, schemas)
                args = [args[0]] + a
            else:
                args, kwargs = _check_args(args, kwargs, name, schemas)
        value = decoratee(*args, **kwargs)
        with util.exceptions.update('schema.check failed for return value of function:\n {}'.format(name), AssertionError):
            output = validate(schemas['returns'], value)
        return output
    decorated.__orig_code__ = decoratee.__code__
    return decorated

def _gen_check(decoratee, name, schemas):
    @functools.wraps(decoratee)
    def decorated(*args, **kwargs):
        with util.exceptions.update('schema.check failed for generator:\n  {}'.format(name), AssertionError, when=lambda x: 'failed for ' not in x):
            if args and decoratee.__code__ is getattr(getattr(args[0], decoratee.__name__, None), '__orig_code__', None):
                a, kwargs = _check_args(args[1:], kwargs, name, schemas)
                args = [args[0]] + a
            else:
                args, kwargs = _check_args(args, kwargs, name, schemas)
        generator = decoratee(*args, **kwargs)
        to_send = None
        first_send = True
        send_exception = False
        while True:
            if not first_send:
                with util.exceptions.update('schema.check failed for send value of generator:\n {}'.format(name), AssertionError):
                    to_send = validate(schemas['sends'], to_send)
            first_send = False
            try:
                if send_exception:
                    to_yield = generator.throw(*send_exception)
                    send_exception = False
                else:
                    to_yield = generator.send(to_send)
                with util.exceptions.update('schema.check failed for yield value of generator:\n {}'.format(name), AssertionError):
                    to_yield = validate(schemas['yields'], to_yield)
            except StopIteration as e:
                with util.exceptions.update('schema.check failed for return value of generator:\n {}'.format(name), AssertionError):
                    return validate(schemas['returns'], getattr(e, 'value', None))
            try:
                to_send = yield to_yield
            except:
                send_exception = sys.exc_info()
    decorated.__orig_code__ = decoratee.__code__
    return decorated

def _coroutine_check(decoratee, name, schemas):
    @functools.wraps(decoratee)
    async def decorated(*args, **kwargs):
        with util.exceptions.update('schema.check failed for coroutine:\n  {}'.format(name), AssertionError, when=lambda x: 'failed for ' not in x):
            if args and decoratee.__code__ is getattr(getattr(args[0], decoratee.__name__, None), '__orig_code__', None):
                a, kwargs = _check_args(args[1:], kwargs, name, schemas)
                args = [args[0]] + a
            else:
                args, kwargs = _check_args(args, kwargs, name, schemas)
            val = await decoratee(*args, **kwargs)
            return validate(schemas['returns'], val)
    decorated.__orig_code__ = decoratee.__code__
    return decorated

# TODO schema.check doesnt support switching between arg and kwarg at call time.
# u have to use which ever way you defined the annotation. ie default value?
# or actually is this a feature? helpful constraint?
@util.func.optionally_parameterized_decorator
def check(*args, **kwargs):
    # TODO add doctest with :fn and args/kwargs
    def decorator(decoratee):
        if disabled:
            return decoratee
        name = util.func.name(decoratee)
        schemas = _get_schemas(decoratee, args, kwargs)
        if inspect.iscoroutinefunction(decoratee):
            decorated = _coroutine_check(decoratee, name, schemas)
        elif inspect.isgeneratorfunction(decoratee):
            decorated = _gen_check(decoratee, name, schemas)
        else:
            decorated = _fn_check(decoratee, name, schemas)
        decorated._schema = schemas['arg'], {k: v for k, v in list(schemas['kwarg'].items()) + [['returns', schemas['returns']]]}
        return decorated
    return decorator
