# py-schema

### data centric schema validation

### installation
```
git clone https://github.com/nathants/py-schema
cd py-schema
pip install -r requirements.txt .
```

### for better performance, disable schemas

```
SCHEMA_DISABLE=y python server.py
```

note: you must not rely on optional value behavior if you disable schemas, instead use `dict.get()`
