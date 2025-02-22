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

## Session State

Session State is a dict-like interface that can be used to save info that is preserved between script runs

This could be

- images of the rink
- images of the team logos?
- a dict of demo games?

Sessions can also store connections! This can be done like this:

```python
import streamlit as st

conn = st.connection("my_database")
df = conn.query("select * from my_table")
st.dataframe(df)

```
