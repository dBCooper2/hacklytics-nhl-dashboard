## Caching

You can cache functions in streamlit with the following command:

```python
@st.cache_data
def long_running_function(param1, param2):
    return …
```

This can be done to store...

- Python Primitives (int, str, etc.)
- DataFrames
- API Calls

You cannot store...

- ML Models
- Database Connections