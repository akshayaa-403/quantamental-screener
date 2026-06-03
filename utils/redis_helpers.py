import pandas as pd
import io

def serialize_dataframe(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    return buffer.getvalue()

def deserialize_dataframe(data: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(data)
    return pd.read_parquet(buffer)